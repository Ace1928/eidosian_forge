import os
import sqlite3
import hashlib
from sentence_transformers import SentenceTransformer
from tkinter import (
    filedialog,
    Tk,
    Button,
    Label,
    Entry,
    messagebox,
    Frame,
    Listbox,
    Toplevel,
    Text,
    Scrollbar,
    VERTICAL,
    HORIZONTAL,
)
from tkinter import ttk
import pdfplumber  # Modern library for PDF text extraction
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging  # For detailed logging throughout the module
import docx2txt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd  # For advanced data manipulation and analysis
import PyPDF2  # Additional modern library for PDF processing
import mimetypes  # For determining file types
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph, draw_networkx
import json
import io
from PIL import Image, ImageTk

nltk.download("punkt")
nltk.download("stopwords")

# Initialize the Sentence Transformer Model with a specific pre-trained model.
model = SentenceTransformer("all-mpnet-base-v2")

# Establish a connection to the SQLite databases for storing embeddings and analysis.
conn_embeddings = sqlite3.connect("embeddings.db")
conn_analysis = sqlite3.connect("analysis.db")
c_embeddings = conn_embeddings.cursor()
c_analysis = conn_analysis.cursor()

# Create tables in the databases if they do not already exist.
c_embeddings.execute(
    """CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, embeddings BLOB, filename TEXT)"""
)
c_analysis.execute(
    """CREATE TABLE IF NOT EXISTS analysis (hash TEXT PRIMARY KEY, word_count INTEGER, most_common_words TEXT, clusters TEXT, wordcloud BLOB, knowledge_graph BLOB)"""
)
conn_embeddings.commit()
conn_analysis.commit()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def hash_file(filepath: str) -> str:
    """
    Generate a SHA-256 hash for a file to uniquely identify it.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_text_from_file(filepath: str) -> str:
    """
    Extract text from a file using various libraries based on file format.
    """
    try:
        file_type, _ = mimetypes.guess_type(filepath)
        if file_type == "application/pdf":
            with pdfplumber.open(filepath) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
            return " ".join(filter(None, pages))
        elif file_type in ["text/plain", "text/markdown"]:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read()
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]:
            return docx2txt.process(filepath)
        else:
            logging.error(f"Unsupported file format for {filepath}")
            return None
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        raise


def store_embeddings(text: str, file_hash: str, filename: str) -> None:
    """
    Generate embeddings for the provided text and store them in the database.
    """
    embeddings = model.encode([text], show_progress_bar=True)
    c_embeddings.execute(
        "INSERT OR IGNORE INTO documents (hash, embeddings, filename) VALUES (?, ?, ?)",
        (file_hash, embeddings.tobytes(), filename),
    )
    conn_embeddings.commit()


def analyze_text(text: str, file_hash: str) -> None:
    """
    Perform advanced text analysis using NLTK and store the results in the database.
    """
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    filtered_words = [word for word in tokens if word not in stopwords.words("english")]
    word_freq = nltk.FreqDist(filtered_words)
    most_common_words = str(word_freq.most_common(10))
    word_count = len(tokens)

    # Perform clustering on the embeddings
    embeddings = model.encode(sentences)
    kmeans = KMeans(n_clusters=5).fit(embeddings)
    agglomerative = AgglomerativeClustering(n_clusters=5).fit(embeddings)
    clusters = {
        "kmeans": kmeans.labels_.tolist(),
        "agglomerative": agglomerative.labels_.tolist(),
    }

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400).generate(text)
    wordcloud_image = wordcloud.to_image()
    wordcloud_image_bytes = io.BytesIO()
    wordcloud_image.save(wordcloud_image_bytes, format="PNG")

    # Generate Knowledge Graph
    graph = Graph()
    for sentence in sentences:
        graph.add_nodes_from(nltk.word_tokenize(sentence))
        edges = list(
            zip(nltk.word_tokenize(sentence), nltk.word_tokenize(sentence)[1:])
        )
        graph.add_edges_from(edges)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="#FF5733",
        node_size=1500,
        font_size=10,
    )
    plt.savefig("knowledge_graph.png")
    plt.close()

    c_analysis.execute(
        "INSERT OR IGNORE INTO analysis (hash, word_count, most_common_words, clusters, wordcloud, knowledge_graph) VALUES (?, ?, ?, ?, ?, ?)",
        (
            file_hash,
            word_count,
            most_common_words,
            str(clusters),
            wordcloud_image_bytes.getvalue(),
            json.dumps(pos),
        ),
    )
    conn_analysis.commit()


def process_folder(folder_path: str) -> None:
    """
    Process all files in the specified folder, generating and storing embeddings and analysis for each file.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_hash = hash_file(filepath)
            c_embeddings.execute(
                "SELECT hash FROM documents WHERE hash = ?", (file_hash,)
            )
            if c_embeddings.fetchone() is None:
                try:
                    text = extract_text_from_file(filepath)
                    if text:
                        store_embeddings(text, file_hash, file)
                        analyze_text(text, file_hash)
                    else:
                        logging.info(f"Failed to process {filepath}")
                except Exception as e:
                    logging.error(f"Error processing file {filepath}: {e}")
                    continue


def select_folder() -> None:
    """
    Create a graphical user interface (GUI) to allow the user to select a folder for processing.
    """
    root = Tk()
    root.title("Document Processor")

    Label(root, text="Select a folder to process:").pack()
    Button(root, text="Select Folder", command=lambda: process_and_close(root)).pack()
    root.mainloop()


def process_and_close(root: Tk) -> None:
    """
    Handle the folder selection and initiate the processing of the selected folder.
    """
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        process_folder(folder_selected)
        messagebox.showinfo("Process Complete", "All files have been processed.")
    root.destroy()


def load_embeddings() -> tuple:
    """
    Load all embeddings and filenames from the database.
    """
    c_embeddings.execute("SELECT embeddings, filename FROM documents")
    data = c_embeddings.fetchall()
    embeddings = [np.frombuffer(d[0], dtype=np.float32) for d in data]
    filenames = [d[1] for d in data]
    return embeddings, filenames


def search_embeddings(query: str, num_results: int = 5) -> list:
    """
    Search for documents similar to a given query based on cosine similarity of embeddings.
    """
    query_embedding = model.encode([query])[0]
    embeddings, filenames = load_embeddings()
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    return [(filenames[i], similarities[i]) for i in top_indices]


def show_results(query: str, root: Tk) -> None:
    """
    Display the search results in a new window.
    """
    results = search_embeddings(query)
    result_window = Toplevel(root)
    result_window.title("Search Results")

    for filename, sim in results:
        Label(result_window, text=f"{filename}: {sim:.2f} similarity").pack()

    result_window.mainloop()


def update_document_list(listbox: Listbox) -> None:
    """
    Update the document list in the Document List tab.
    """
    c_embeddings.execute("SELECT filename FROM documents")
    files = c_embeddings.fetchall()
    listbox.delete(0, "end")
    for file in files:
        listbox.insert("end", file[0])


def display_wordcloud(tab: ttk.Frame) -> None:
    """
    Display the wordcloud in the WordCloud tab.
    """
    c_analysis.execute("SELECT wordcloud FROM analysis")
    wordcloud_data = c_analysis.fetchall()
    for data in wordcloud_data:
        image = Image.open(io.BytesIO(data[0]))
        photo = ImageTk.PhotoImage(image)
        label = Label(tab, image=photo)
        label.image = photo  # keep a reference!
        label.pack()


def display_knowledge_graph(tab: ttk.Frame) -> None:
    """
    Display the knowledge graph in the Knowledge Graph tab.
    """
    c_analysis.execute("SELECT knowledge_graph FROM analysis")
    graph_data = c_analysis.fetchall()
    for data in graph_data:
        pos = json.loads(data[0])
        graph = Graph()
        graph.add_nodes_from(pos.keys())
        for node, edges in pos.items():
            for edge in edges:
                graph.add_edge(node, edge)
        plt.figure(figsize=(10, 10))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            edge_color="#FF5733",
            node_size=1500,
            font_size=10,
        )
        plt.savefig("knowledge_graph_display.png")
        plt.close()
        image = Image.open("knowledge_graph_display.png")
        photo = ImageTk.PhotoImage(image)
        label = Label(tab, image=photo)
        label.image = photo  # keep a reference!
        label.pack()


def main_gui() -> None:
    """
    Create the main graphical user interface for searching documents.
    """
    root = Tk()
    root.title("Search Documents")

    tab_control = ttk.Notebook(root)

    # Search tab
    search_tab = ttk.Frame(tab_control)
    tab_control.add(search_tab, text="Search")
    Label(search_tab, text="Enter search query:").pack()
    query_input = Entry(search_tab, width=50)
    query_input.pack()
    Button(
        search_tab, text="Search", command=lambda: show_results(query_input.get(), root)
    ).pack()

    # WordCloud tab
    wordcloud_tab = ttk.Frame(tab_control)
    tab_control.add(wordcloud_tab, text="WordCloud")
    display_wordcloud(wordcloud_tab)

    # Knowledge Graph tab
    graph_tab = ttk.Frame(tab_control)
    tab_control.add(graph_tab, text="Knowledge Graph")
    display_knowledge_graph(graph_tab)

    # Document List tab
    doc_list_tab = ttk.Frame(tab_control)
    tab_control.add(doc_list_tab, text="Document List")
    listbox = Listbox(doc_list_tab)
    listbox.pack(fill="both", expand=True)
    update_document_list(listbox)

    tab_control.pack(expand=1, fill="both")

    root.mainloop()


if __name__ == "__main__":
    select_folder()
    main_gui()
