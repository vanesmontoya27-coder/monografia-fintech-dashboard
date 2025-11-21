import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import kruskal, spearmanr
import os

st.set_page_config(page_title="Análisis Fintech Colombia", layout="wide")
template_style = "simple_white"


@st.cache_data
def get_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    df.dropna(how="all", inplace=True)

    renames = {
        "11. En general, ¿qué tan probable es que en los próximos 12 meses continúes usando o recomiendes a otras personas las billeteras o créditos digitales?": "y_intencion",
        "5. En general, ¿qué tan seguro(a) te sientes al usar servicios Fintech para manejar tu dinero o tus datos personales?": "x_seguridad",
        "12. ¿Cómo calificas tu nivel de conocimiento o educación financiera para manejar servicios Fintech?": "x_educacion_fin",
        "8.  ¿Cuáles de los siguientes factores influyen más en tu confianza o desconfianza hacia las plataformas Fintech?": "txt_factores",
        "13. Selecciona tu rango de edad:": "d_edad",
        "15. Selecciona tu género": "d_genero",
        " 1. En relación con el servicio Nequi, selecciona la opción que te describa mejor:": "uso_nequi",
        "2. En relación con el servicio Daviplata, selecciona la opción que te describa mejor:": "uso_daviplata",
        " 3. En relación con el servicio Addi, selecciona la opción que te describa mejor:": "uso_addi",
        "4. En relación con el servicio Sistecrédito, selecciona la opción que te describa mejor:": "uso_sistecredito",
        "9. ¿Has tenido alguna experiencia negativa al usar billeteras o créditos digitales (errores, fraudes, demoras, mala atención)?": "exp_negativa",
        "10. En tu opinión, ¿las plataformas Fintech ofrecen una protección de datos personales igual, mejor o peor que la banca tradicional?": "comp_banca",
    }

    df.columns = df.columns.str.strip()
    df.rename(columns={k.strip(): v for k, v in renames.items()}, inplace=True)

    map_seguridad = {
        "Nada seguro(a).": 1,
        "Poco seguro(a)": 2,
        "Poco seguro(a).": 2,
        "Neutral": 3,
        "Neutral.": 3,
        "Seguro(a).": 4,
        "Muy seguro(a).": 5,
    }

    map_intencion = {
        "Nada probable.": 1,
        "Poco probable.": 2,
        "Indiferente.": 3,
        "Neutral": 3,
        "Neutral.": 3,
        "Probable.": 4,
        "Muy probable.": 5,
    }

    map_edu = {"Nulo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}

    df["num_seguridad"] = df["x_seguridad"].map(map_seguridad)
    df["num_intencion"] = df["y_intencion"].map(map_intencion)
    df["num_educacion"] = df["x_educacion_fin"].map(map_edu)

    if "txt_factores" in df.columns:
        factors_list = df["txt_factores"].dropna().str.split(",").tolist()
        flat_list = [
            item.strip().rstrip(".") for sublist in factors_list for item in sublist
        ]
        unique_factors = list(set(flat_list))

        for factor in unique_factors:
            if len(factor) > 5:
                df[f"factor_{factor}"] = df["txt_factores"].apply(
                    lambda x: 1 if isinstance(x, str) and factor in x else 0
                )

    return df


st.title("Análisis de Determinantes en la Adopción Fintech")
st.markdown(
    """
**Tablero de Control Profesional - Trabajo de Grado**
Este sistema permite explorar interactivamente las relaciones estadísticas entre la percepción de seguridad, 
el perfil demográfico y la decisión de adopción de tecnologías financieras en Colombia.
"""
)

ARCHIVO_POR_DEFECTO = "datos.csv"

df = None

if os.path.exists(ARCHIVO_POR_DEFECTO):
    try:
        df = get_data(ARCHIVO_POR_DEFECTO)
        st.success(f"Datos cargados automáticamente desde: {ARCHIVO_POR_DEFECTO}")
    except Exception as e:
        st.error(f"Error al cargar el archivo local: {e}")

if df is None:
    st.warning(
        f"No se encontró el archivo '{ARCHIVO_POR_DEFECTO}'. Por favor cárgalo manualmente."
    )
    uploaded_file = st.sidebar.file_uploader("Cargar Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = get_data(uploaded_file)

if df is not None:

    tab1, tab2, tab3 = st.tabs(
        ["Análisis Descriptivo", "Inferencia Estadística", "Modelado Predictivo"]
    )

    with tab1:
        st.header("Panorama General de la Muestra y Adopción")

        st.subheader("1. Caracterización Demográfica")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            df_edad = df["d_edad"].value_counts().reset_index()
            df_edad.columns = ["Rango de Edad", "Frecuencia"]
            
            
            total_n = df_edad["Frecuencia"].sum()
            df_edad["Porcentaje"] = (df_edad["Frecuencia"] / total_n * 100).round(1).astype(str) + '%'

            fig_edad = px.bar(
                df_edad,
                x="Rango de Edad",
                y="Frecuencia",
                text="Porcentaje", 
                title="Distribución por Rango de Edad",
                color="Rango de Edad",
                color_discrete_sequence=px.colors.qualitative.Safe,
                template=template_style,
            )
            
            max_val = df_edad["Frecuencia"].max()
            fig_edad.update_yaxes(range=[0, max_val * 1.2]) 
            fig_edad.update_traces(textposition='outside', textfont_size=14)
            fig_edad.update_layout(showlegend=False)
            
            st.plotly_chart(fig_edad, use_container_width=True)

        with col_d2:
            fig_gen = px.pie(
                df,
                names="d_genero",
                title="Composición por Género",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template=template_style,
                hole=0.4,
            )
            st.plotly_chart(fig_gen, use_container_width=True)

        st.markdown("---")

        st.subheader("2. Uso de Herramientas Fintech")
        st.markdown("Comparativa de usuarios activos ('Lo uso actualmente') respecto al total de encuestados.")

        tools = ["uso_nequi", "uso_daviplata", "uso_addi", "uso_sistecredito"]
        tool_names = ["Nequi", "Daviplata", "Addi", "Sistecrédito"]
        adoption_rates = []

        for tool in tools:
            if tool in df.columns:
                count = df[
                    df[tool]
                    .astype(str)
                    .str.contains("actualmente", case=False, na=False)
                ].shape[0]
                adoption_rates.append(count)
            else:
                adoption_rates.append(0)

        df_adoption = pd.DataFrame(
            {"Herramienta": tool_names, "Usuarios Activos": adoption_rates}
        )
        df_adoption = df_adoption.sort_values("Usuarios Activos", ascending=False)

  
        total_personas = len(df)
        df_adoption["Porcentaje"] = (df_adoption["Usuarios Activos"] / total_personas * 100).round(1).astype(str) + '%'

        fig_adopt = px.bar(
            df_adoption,
            x="Herramienta",
            y="Usuarios Activos",
            
            text="Porcentaje", 
            
            color="Herramienta",
            title="Cuota de Mercado en la muestra (Penetración)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template=template_style,
        )
        
        max_val = df_adoption["Usuarios Activos"].max()
        fig_adopt.update_yaxes(range=[0, max_val * 1.25])
        fig_adopt.update_traces(textposition='outside', textfont_size=14)
        
        st.plotly_chart(fig_adopt, use_container_width=True)

        st.markdown("---")

        st.subheader("3. Percepción de Seguridad y Experiencia")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown("**Distribución de Seguridad**")
            
       
            df['num_seguridad'] = pd.to_numeric(df['num_seguridad'], errors='coerce')
            
        
            df_sec_filtered = df[df['num_seguridad'].notna() & (df['num_seguridad'] != 3)].copy()
            
            if not df_sec_filtered.empty:
                df_sec_filtered['num_seguridad'] = df_sec_filtered['num_seguridad'].astype(int)
                
                sec_counts = df_sec_filtered['num_seguridad'].value_counts().reset_index()
                sec_counts.columns = ['Nivel', 'Frecuencia'] 
                
                etiquetas_map = {
                    1: "Muy Bajo",
                    2: "Bajo",
                    4: "Alto",
                    5: "Muy Alto"
                }
                sec_counts['Nombre'] = sec_counts['Nivel'].map(etiquetas_map).fillna("Otro")
                
                total_filtrado = sec_counts["Frecuencia"].sum()
                sec_counts["Porcentaje"] = (sec_counts["Frecuencia"] / total_filtrado * 100).round(1).astype(str) + '%'
                
                sec_counts = sec_counts.sort_values('Nivel')

                fig_sec = px.bar(
                    sec_counts,
                    x="Nombre",
                    y="Frecuencia",
                    text="Porcentaje", 
                    title="Percepción de Seguridad",
                    color_discrete_sequence=["#7F55A1"], 
                    template=template_style,
                )
                
                max_val_sec = sec_counts["Frecuencia"].max()
                fig_sec.update_yaxes(range=[0, max_val_sec * 1.25])
                
                fig_sec.update_xaxes(
                    type='category', 
                    categoryorder='array', 
                    categoryarray=["Muy Bajo", "Bajo", "Alto", "Muy Alto"]
                )
                
                fig_sec.update_traces(textposition='outside', textfont_size=14)
                
                st.plotly_chart(fig_sec, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar la distribución de seguridad.")

        with col_s2:

            st.markdown("**Fintech vs. Banca Tradicional**")

            if "comp_banca" in df.columns:

                df_comp = df["comp_banca"].value_counts().reset_index()

                df_comp.columns = ["Opinión", "Conteo"]

                fig_comp = px.pie(

                    df_comp,

                    names="Opinión",

                    values="Conteo",

                    title="Protección de datos: Comparativa",

                    color_discrete_sequence=px.colors.qualitative.Safe,

                )

                st.plotly_chart(fig_comp, use_container_width=True)
                
        with col_s3:
            st.markdown("**Incidentes Reportados**")
            if "exp_negativa" in df.columns:
                df['tipo_incidente'] = df['exp_negativa'].astype(str).apply(
                    lambda x: 'No' if x.strip().lower().startswith('no') else 'Sí'
                )
                
                df_exp = df['tipo_incidente'].value_counts().reset_index()
                df_exp.columns = ["¿Tuvo Incidente?", "Conteo"]
                
                total_incidentes = df_exp["Conteo"].sum()
                df_exp["Porcentaje"] = (df_exp["Conteo"] / total_incidentes * 100).round(1).astype(str) + '%'
                
                fig_exp = px.bar(
                    df_exp,
                    x="¿Tuvo Incidente?",
                    y="Conteo",
                    
                    text="Porcentaje",
                    
                    title="¿Ha tenido experiencias negativas?",
                    color="¿Tuvo Incidente?",
                    color_discrete_map={'Sí': '#BF092F', 'No': '#66B083'}
                )
                
                max_val_exp = df_exp["Conteo"].max()
                fig_exp.update_yaxes(range=[0, max_val_exp * 1.25]) 
                fig_exp.update_traces(textposition='outside', textfont_size=15) 
                fig_exp.update_layout(showlegend=False)
                
                st.plotly_chart(fig_exp, use_container_width=True)

        st.markdown("---")
        st.markdown("**Factores Determinantes de Confianza**")
        factor_cols = [c for c in df.columns if c.startswith("factor_")]
        
        factor_counts = df[factor_cols].sum().sort_values(ascending=True)
        factor_counts.index = factor_counts.index.str.replace("factor_", "")

    
        total_encuestados = len(df)
        pct_text = (factor_counts / total_encuestados * 100).round(1).astype(str) + '%'

        fig_factors = px.bar(
            x=factor_counts.values,
            y=factor_counts.index,
            orientation="h",
            
            text=pct_text,
            
            color_discrete_sequence=["#B05353"],
            
            title="Desglose de Factores de Confianza",
            template=template_style,
        )

 
        fig_factors.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
        
        fig_factors.update_layout(showlegend=False)
        
        max_val = factor_counts.max()
        fig_factors.update_xaxes(range=[0, max_val * 1.1])

        st.plotly_chart(fig_factors, use_container_width=True)

    with tab2:
        st.subheader("Pruebas de Hipótesis")

        st.markdown("#### 1. ¿Existe correlación entre Seguridad e Intención de Uso?")
        st.write(
            "Se utiliza el coeficiente de **Spearman** dado que las variables son ordinales (no siguen una distribución normal perfecta)."
        )

        df_corr = df[["num_seguridad", "num_intencion", "num_educacion"]].dropna()
        corr, p_value = spearmanr(df_corr["num_seguridad"], df_corr["num_intencion"])

        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Coeficiente Rho de Spearman", f"{corr:.4f}")
        col_res2.metric("Valor P (Significancia)", f"{p_value:.4e}")

        if p_value < 0.05:
            st.info(
                f"**Interpretación:** Existe una relación estadísticamente significativa (p < 0.05). La fuerza de la relación es de {corr:.2f}, lo cual indica una correlación positiva moderada."
            )
        else:
            st.error(
                "Interpretación: No existe evidencia estadística suficiente para afirmar una relación."
            )

        st.markdown("---")

        st.markdown("#### 2. ¿Varía la percepción de seguridad según el Género?")
        st.write(
            "Se aplica la prueba **U de Mann-Whitney** (o Kruskal-Wallis) para comparar las medianas de seguridad entre géneros."
        )

        generos = df["d_genero"].unique()
        groups = [
            df[df["d_genero"] == g]["num_seguridad"].dropna()
            for g in generos
            if isinstance(g, str)
        ]

        if len(groups) > 1:
            stat_k, p_k = kruskal(*groups)
            st.write(f"**Estadístico H:** {stat_k:.4f}, **Valor P:** {p_k:.4f}")
            
            if p_k > 0.05:
                st.info(
                    "💡 **Interpretación:** El Valor P es mayor a 0.05. Esto indica que **NO hay diferencias significativas**. Hombres y mujeres perciben el riesgo de manera similar."
                )
            else:
                st.success(
                    "💡 **Interpretación:** El Valor P es menor a 0.05. **SÍ existen diferencias significativas** en la percepción de seguridad entre géneros."
                )


            df['num_seguridad'] = pd.to_numeric(df['num_seguridad'], errors='coerce')
            df_box_filtered = df[df['num_seguridad'].notna() & (df['num_seguridad'] != 3)].copy()

            fig_box = px.box(
                df_box_filtered, 
                x="d_genero",
                y="num_seguridad",
                color="d_genero",
                template=template_style,
                title="Distribución de Seguridad por Género",
                
 
                color_discrete_sequence=["#7F55A1", "#B05353", "#5D6D7E"], 
            )
            
            fig_box.update_yaxes(
                title="Nivel de Seguridad Percibido",
                tickmode='array',
                tickvals=[1, 2, 4, 5],
                ticktext=["Muy Bajo", "Bajo", "Alto", "Muy Alto"] 
            )
            
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.header("Modelo de Regresión e Interpretación de Impacto")

        st.markdown(
            """
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <strong>¿Qué hace esta sección?</strong><br>
        Aquí usamos matemáticas (Regresión Estadística) para "pesar" las variables. 
        Imagine una balanza: ponemos la <strong>Seguridad</strong> en un lado y la <strong>Educación Financiera</strong> en el otro. 
        El modelo nos dice cuál de las dos inclina más la balanza hacia la decisión de usar la App.
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.write("")

        X = df[["num_seguridad", "num_educacion"]].dropna()

        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = df.loc[X.index, "num_intencion"]

        if not X.empty:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            st.subheader("1. ¿Podemos confiar en este análisis?")

            col_metric1, col_metric2 = st.columns(2)
            r2 = model.rsquared
            p_f = model.f_pvalue

            with col_metric1:
                st.metric(label="Poder Explicativo (R²)", value=f"{r2:.1%}")
                st.info(
                    f"""
                **Lectura Fácil:** Este modelo explica el **{r2:.1%}** del comportamiento de los usuarios.
                El porcentaje restante depende de cosas que no preguntamos (como tasas de interés, publicidad, necesidad urgente, etc.).
                """
                )

            with col_metric2:
                st.metric(label="Certeza Estadística (Valor P)", value=f"{p_f:.4f}")
                if p_f < 0.05:
                    st.success(
                        "**Veredicto:** El análisis es SÓLIDO. Los resultados no son producto de la suerte."
                    )
                else:
                    st.error(
                        "**Veredicto:** Cuidado. Los datos no son suficientes para sacar conclusiones definitivas."
                    )

            st.markdown("---")

            st.subheader("2. ¿Qué pesa más en la decisión?")
            st.markdown(
                """
            El siguiente gráfico muestra la **fuerza** de cada variable.
            * **Barra Larga:** Influye mucho.
            * **Barra Verde:** Es un dato confirmado estadísticamente.
            * **Barra Gris:** El dato es incierto (podría ser casualidad).
            """
            )

            nombres_amigables = {
                "const": "Constante (Base)",
                "num_seguridad": "Percepción de Seguridad",
                "num_educacion": "Educación Financiera",
            }
            variables_modelo = [nombres_amigables.get(v, v) for v in model.params.index]

            results_df = pd.DataFrame(
                {
                    "Variable": variables_modelo,
                    "Impacto": model.params.values,
                    "Valor P": model.pvalues.values,
                }
            )

            plot_df = results_df[results_df["Variable"] != "Constante (Base)"].copy()
            plot_df["Confiabilidad"] = plot_df["Valor P"].apply(
                lambda x: (
                    "Datos Confiables (Significativo)"
                    if x < 0.05
                    else "Datos Inciertos (No Sig.)"
                )
            )

            fig_coef = px.bar(
                plot_df,
                x="Impacto",
                y="Variable",
                color="Confiabilidad",
                orientation="h",
                text_auto=".2f",
                color_discrete_map={
                    "Datos Confiables (Significativo)": "#2ecc71",
                    "Datos Inciertos (No Sig.)": "#bdc3c7",
                },
                title="Peso de cada factor en la Intención de Uso",
            )

            fig_coef.update_layout(
                xaxis_title="Puntos de Aumento en la Intención de Uso",
                yaxis_title="",
                legend_title="Calidad del Dato",
                template="simple_white",
                font=dict(size=14),
            )
            st.plotly_chart(fig_coef, use_container_width=True)

            try:
                coef_seg = model.params["num_seguridad"]

                st.markdown(
                    f"""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                    <strong>Conclusión para el Lector:</strong><br>
                    Por cada punto que usted logre aumentar la <b>Seguridad Percibida</b> en sus usuarios, 
                    la probabilidad de que usen su servicio aumentará en <b>{coef_seg:.2f} puntos</b>.
                    Es decir, invertir en seguridad tiene un retorno directo en la adopción.
                </div>
                """,
                    unsafe_allow_html=True,
                )
            except:
                pass

            st.markdown("---")

            st.subheader("3. ¿Qué hace que la gente se sienta segura?")
            st.markdown(
                "Analizamos la correlación estadística: ¿Qué palabras clave están más asociadas a decir 'Me siento seguro'?"
            )

            factor_cols = [c for c in df.columns if c.startswith("factor_")]
            correlations = {}

            if len(factor_cols) > 0:
                for col in factor_cols:
                    corr_val = df[col].corr(df["num_seguridad"])
                    clean_name = col.replace("factor_", "")
                    correlations[clean_name] = corr_val

                df_corr_factors = pd.DataFrame.from_dict(
                    correlations, orient="index", columns=["Correlación"]
                ).reset_index()

                df_corr_factors.columns = ["Factor", "Correlación"]

                df_corr_factors = df_corr_factors.sort_values(
                    by="Correlación", ascending=True
                )

                fig_factors = px.bar(
                    df_corr_factors,
                    x="Correlación",
                    y="Factor",
                    orientation="h",
                    color="Factor",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    title="Impacto de cada característica en la sensación de Seguridad",
                )

                fig_factors.add_vline(x=0, line_width=2, line_color="#333333")

                fig_factors.update_yaxes(
                    categoryorder="array", categoryarray=df_corr_factors["Factor"]
                )

                fig_factors.update_layout(
                    xaxis_title="Relación con la Seguridad (Negativa < 0 > Positiva)",
                    yaxis_title="",
                    template="simple_white",
                    showlegend=False,
                    font=dict(size=12),
                )
                st.plotly_chart(fig_factors, use_container_width=True)

                st.info(
                    """
                **Nota de Lectura:** * Barras hacia la **Derecha**: Factores que generan confianza (Positivos).
                * Barras hacia la **Izquierda**: Factores asociados a desconfianza (Negativos).
                """
                )
        else:
            st.error("No hay suficientes datos válidos para generar el modelo.")
else:
    st.info("Esperando archivo CSV para iniciar análisis profesional.")
