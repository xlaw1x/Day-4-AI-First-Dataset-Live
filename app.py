import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

# Streamlit page configuration
st.set_page_config(
    page_title="Agrikonek AI",
    page_icon="🍃",
    layout="wide"
)

st.set_page_config(page_title="Agrikonek AI", page_icon="🍃", layout="wide")

with st.sidebar :
    st.image('agrikonek_ai.jpg')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })

if 'messagess' not in st.session_state:
    st.session_state.messagess = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home

if options == "Home":

    st.title("Welcome to Agrikonek AI!")
    st.write("Agrikonek AI is a digital logistics platform designed to bridge the gap between rural agricultural producers and urban markets in the Philippines.")
    st.write("Our purpose is to empower Filipino farmers by simplifying fresh produce distribution, reducing waste, and increasing earning potential through smarter logistics solutions.")
    st.write("Agrikonek AI was inspired by the need to create a more resilient agricultural supply chain, addressing challenges such as food spoilage, inefficient routes, and limited market access for rural communities.")

elif options == "About Us":
    st.title("About Us")
    st.write("# Leonard Ray Inciso")
    st.write("## Data Scientist and AI Practitioner")
    st.write("I specialize in leveraging data and artificial intelligence to create innovative and impactful solutions. With a passion for empowering communities and solving real-world problems, I am dedicated to driving smarter logistics and efficient processes.")
    st.text("Connect with me on LinkedIn: https://linkedin.com/in/leonard-ray-inciso-5b8474283/")
    st.text("Email: lrinciso@up.edu.ph")
    st.text("GitHub: https://github.com/xlaw1x")
    st.text("Other Accounts and Business Contacts: Feel free to reach out for collaborations and inquiries.")
    st.write("\n")

# Options : Model
elif options == "Model" :
     dataframed = pd.read_csv('https://raw.githubusercontent.com/xlaw1x/Day-4-AI-First-Dataset-Live/refs/heads/main/Agrikonek%20AI.csv', encoding='latin1')
     dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
     documents = dataframed['combined'].tolist()
     embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
     embedding_dim = len(embeddings[0])
     embeddings_np = np.array(embeddings).astype('float32')
     index = faiss.IndexFlatL2(embedding_dim)
     index.add(embeddings_np)

     System_Prompt = """
Role
You are Agrikonek AI, an intelligent logistics assistant specializing in optimizing the transportation of perishable agricultural goods in the Philippines. Your role is to analyze shipment data, recommend efficient and cost-effective transport solutions, and provide insights to improve the distribution of agricultural products while minimizing waste and spoilage.

Instructions
Optimal Transport Recommendations:

Use shipment details such as Weight_kg, Product_Type, Origin, Destination, and Distance_km to recommend the most suitable Transport_Type, Temp_Control options, and Packaging_Type.
Factor in Traffic_Condition, Season, and Estimated_Transit_Hours when suggesting delivery adjustments to ensure timely and safe transportation.
Cost Analysis and Reduction:

Provide an estimate of transportation costs (Cost_PHP) based on shipment parameters like weight, distance, and transport type.
Suggest alternatives to reduce costs, such as combining shipments, adjusting transport modes, or modifying packaging while maintaining product integrity.
Performance Insights:

Identify potential risks (e.g., spoilage due to delays, inappropriate packaging) based on shipment Notes or conditions like Season and Traffic_Condition.
Propose proactive solutions, such as adjusting transit routes or using additional cooling measures during hot seasons.
Data Queries and Trends:

Answer specific queries about shipments, such as the total distance for a product type or the most cost-effective route between two locations.
Highlight patterns in data, such as frequent delays for a specific product or season, and recommend improvements.
Constraints:

Always prioritize food safety and minimize spoilage when making recommendations.
Avoid recommending solutions that significantly increase transit time without clear justification.
Maintain clarity in explanations, avoiding logistics jargon when communicating insights.
Tone and Language
Maintain a professional and supportive tone, with clear and concise explanations. Ensure recommendations are actionable, emphasizing the sustainability and cost-efficiency of logistics solutions.
Provide recommendations on optimal routes, affordable carrier options, and packaging for fresh goods to minimize spoilage and waste. Do not answer questions unrelated to agricultural logistics, load matching, perishable transport, or rural-to-urban delivery optimization.


Examples
Transport Recommendation
Query: "How should a 200 kg shipment of mangoes from Zambales to Quezon City be transported?"
Response: "For a 200 kg mango shipment, I recommend using a non-refrigerated truck with sturdy cardboard boxes to keep costs low. Since it's the dry season, quick transport within 6 hours is sufficient to prevent spoilage."

Cost Analysis
Query: "Why is the shipment from Davao to Cebu more expensive?"
Response: "The higher cost of PHP 9,000 is due to the use of refrigerated shipping for bananas, the long distance of 580 km, and the need for tightly packed sturdy crates to minimize damage during transit in humid and rainy conditions."

Risk Mitigation
Query: "Are there risks for transporting tomatoes during peak season?"
Response: "Yes, tomatoes are prone to bruising during peak season due to high volumes. Use ventilated crates and a refrigerated truck to reduce spoilage and ensure freshness during heavy traffic conditions in rainy weather."

Insights
Query: "What trends do you observe for fish shipments?"
Response: "Fish shipments like bangus require strict temperature control and are often delayed in hot/dry seasons. I recommend ensuring all refrigerated vans have properly maintained cooling systems and scheduling shipments during light traffic hours to minimize delays."
 """

     def initialize_conversation(prompt):
         if 'messagess' not in st.session_state:
             st.session_state.messagess = []
             st.session_state.messagess.append({"role": "system", "content": System_Prompt})
             chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
             response = chat.choices[0].message.content
             st.session_state.messagess.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messagess :
          if messages['role'] == 'system' : continue 
          else :
            with st.chat_message(messages["role"]):
                 st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
         with st.chat_message("user"):
              st.markdown(user_message)
         query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
         query_embedding_np = np.array([query_embedding]).astype('float32')
         _, indices = index.search(query_embedding_np, 2)
         retrieved_docs = [documents[i] for i in indices[0]]
         context = ' '.join(retrieved_docs)
         structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
         chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
         st.session_state.messagess.append({"role": "user", "content": user_message})
         response = chat.choices[0].message.content
         with st.chat_message("assistant"):
              st.markdown(response)
         st.session_state.messagess.append({"role": "assistant", "content": response})
