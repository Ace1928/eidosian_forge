import os
import torch
import logging
import streamlit as st
import duckdb
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

logging.basicConfig(level=logging.INFO)
# Streamlit app setup
st.set_page_config(page_title="Interactive Text Generation with Phi-3", layout="wide")

# Constants for paths
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MODEL_PATH = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)
DB_PATH = "/home/lloyd/Downloads/local_model_store/phi3t5_conversation_embeddings.db"
EMBEDDING_MODEL_NAME = "t5-small"
EMBEDDING_MODEL_PATH = "/home/lloyd/Downloads/local_model_store/t5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize database connection
conn = duckdb.connect(database=DB_PATH, read_only=False)
conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_id START 1")
conn.execute(
    "CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY DEFAULT nextval('seq_id'), input TEXT, output TEXT, input_embedding BLOB, output_embedding BLOB)"
)

## Check if the model and tokenizer are already downloaded
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = T5ForConditionalGeneration.from_pretrained(EMBEDDING_MODEL_NAME)
    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    embedding_tokenizer.save_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model.save_pretrained(EMBEDDING_MODEL_PATH)
    logging.info("Model and tokenizer downloaded and saved.")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logging.info("Model and tokenizer loaded from local storage.")

# Set device based on your hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
embedding_model.to(device)
logging.info(f"Model moved to device: {device}")

# Generate text
prompt = "Once upon a time, in a far-off land"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
logging.info("Input encoded and moved to device.")

try:
    outputs = model.generate(
        inputs,
        max_new_tokens=500,
        top_p=0.95,
        do_sample=True,
        top_k=60,
        temperature=0.95,
        early_stopping=True,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Text generation successful.")
except Exception as e:
    logging.error(f"Error during text generation: {e}")
    generated_text = "Error in text generation."

print("Generated Text:", generated_text)


model.to(device)
embedding_model.to(device)
logging.info(f"Models moved to device: {device}")

st.title("Interactive Text Generation with Phi-3")

# Create two columns for chat interface
col1, col2 = st.columns([2, 1])

# Conversation history
conversation_history = conn.execute("SELECT input, output FROM embeddings").fetchall()

with col2:
    st.header("Conversation Log")
    conversation_container = st.empty()

# Display conversation history
with conversation_container.container():
    for input_text, output_text in conversation_history:
        st.markdown(f"**User:** {input_text}")
        st.markdown(f"**Assistant:** {output_text}")
        st.markdown("---")


# Text generation function
@st.cache_resource
def generate_text(prompt: str):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logging.info("Input encoded and moved to device.")
    try:
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            top_p=0.95,
            do_sample=True,
            top_k=60,
            temperature=0.95,
            early_stopping=True,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Text generation successful.")
        return generated_text
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        return "Error in text generation."


# Embedding generation function
@st.cache_resource
def generate_embeddings(text: str):
    input_ids = embedding_tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = (
            embedding_model.encoder(input_ids)
            .last_hidden_state.mean(dim=1)
            .cpu()
            .numpy()
        )
    return embeddings


# Streamlit interaction
with col1:
    with st.form("text_generation_form"):
        user_input = st.text_input("Enter your prompt:", key="user_input")
        submitted = st.form_submit_button("Submit")
        if submitted and user_input:
            generated_text = generate_text(user_input)
            st.markdown(f"**Assistant:** {generated_text}")

            # Embedding and storing the conversation
            input_embeddings = generate_embeddings(user_input)
            output_embeddings = generate_embeddings(generated_text)

            conn.execute(
                "INSERT INTO embeddings (input, output, input_embedding, output_embedding) VALUES (?, ?, ?, ?)",
                [
                    user_input,
                    generated_text,
                    input_embeddings.tobytes(),
                    output_embeddings.tobytes(),
                ],
            )
            logging.info("Conversation embeddings stored.")

            # Update conversation history
            conversation_container.empty()
            with conversation_container.container():
                for input_text, output_text in conn.execute(
                    "SELECT input, output FROM embeddings"
                ).fetchall():
                    st.markdown(f"**User:** {input_text}")
                    st.markdown(f"**Assistant:** {output_text}")
                    st.markdown("---")

# To run on local machine use the following:
# streamlit run /home/lloyd/Downloads/PythonScripts/localphi3llmtextgeneration.py --server.port 8501 --server.address 0.0.0.0
