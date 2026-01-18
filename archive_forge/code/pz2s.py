import os
import sqlite3
import hashlib
from sentence_transformers import SentenceTransformer
from tkinter import filedialog, Tk, Button, Label, Entry, messagebox
import pdfplumber  # Modern library for PDF text extraction
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging  # For detailed logging throughout the module

# Initialize the Sentence Transformer Model with a specific pre-trained model.
model = SentenceTransformer("all-mpnet-base-v2")

# Establish a connection to the SQLite database and create a cursor for executing SQL commands.
conn = sqlite3.connect("embeddings.db")
c = conn.cursor()

# Create a table named 'documents' in the database if it does not already exist.
# This table will store document hashes, embeddings as BLOBs, and filenames.
c.execute(
    """CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, embeddings BLOB, filename TEXT)"""
)
conn.commit()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def hash_file(filepath: str) -> str:
    """
    Generate a SHA-256 hash for a file to uniquely identify it.

    Parameters:
    - filepath: str - The path to the file whose hash is to be generated.

    Returns:
    - str - The hexadecimal string representation of the hash.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_text_from_file(filepath: str) -> str:
    """
    Extract text from a file using the pdfplumber library, which supports PDF file formats.
    For other formats, additional libraries and conditions can be implemented.

    Parameters:
    - filepath: str - The path to the file from which text is to be extracted.

    Returns:
    - str - The extracted text, or None if an error occurs.
    """
    try:
        if filepath.endswith(".pdf"):
            with pdfplumber.open(filepath) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
            return " ".join(filter(None, pages))
        else:
            logging.error(f"Unsupported file format for {filepath}")
            return None
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None


def store_embeddings(text: str, file_hash: str, filename: str) -> None:
    """
    Generate embeddings for the provided text using the SentenceTransformer model and store them in the database.

    Parameters:
    - text: str - The text for which embeddings are to be generated.
    - file_hash: str - The hash of the file from which the text was extracted.
    - filename: str - The name of the file.
    """
    embeddings = model.encode([text], show_progress_bar=True)
    c.execute(
        "INSERT OR IGNORE INTO documents (hash, embeddings, filename) VALUES (?, ?, ?)",
        (file_hash, embeddings.tobytes(), filename),
    )
    conn.commit()


def process_folder(folder_path: str) -> None:
    """
    Process all files in the specified folder, generating and storing embeddings for each file.

    Parameters:
    - folder_path: str - The path to the folder to be processed.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_hash = hash_file(filepath)
            c.execute("SELECT hash FROM documents WHERE hash = ?", (file_hash,))
            if c.fetchone() is None:
                text = extract_text_from_file(filepath)
                if text:
                    store_embeddings(text, file_hash, file)
                else:
                    logging.info(f"Failed to process {filepath}")


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

    Parameters:
    - root: Tk - The root window of the GUI.
    """
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        process_folder(folder_selected)
        messagebox.showinfo("Process Complete", "All files have been processed.")
    root.destroy()


def load_embeddings() -> tuple:
    """
    Load all embeddings and filenames from the database.

    Returns:
    - tuple - A tuple containing two lists: one of embeddings and one of filenames.
    """
    c.execute("SELECT embeddings, filename FROM documents")
    data = c.fetchall()
    embeddings = [np.frombuffer(d[0], dtype=np.float32) for d in data]
    filenames = [d[1] for d in data]
    return embeddings, filenames


def search_embeddings(query: str, num_results: int = 5) -> list:
    """
    Search for documents similar to a given query based on cosine similarity of embeddings.

    Parameters:
    - query: str - The query string to search for.
    - num_results: int - The number of results to return.

    Returns:
    - list - A list of tuples containing filenames and their similarity scores.
    """
    query_embedding = model.encode([query])[0]
    embeddings, filenames = load_embeddings()
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    return [(filenames[i], similarities[i]) for i in top_indices]


def main_gui() -> None:
    """
    Create the main graphical user interface for searching documents.
    """
    root = Tk()
    root.title("Search Documents")

    Label(root, text="Enter search query:").pack()
    query_input = Entry(root, width=50)
    query_input.pack()

    Button(
        root, text="Search", command=lambda: show_results(query_input.get(), root)
    ).pack()

    root.mainloop()


def show_results(query: str, root: Tk) -> None:
    """
    Display the search results in a new window.

    Parameters:
    - query: str - The search query.
    - root: Tk - The root window of the main GUI.
    """
    results = search_embeddings(query)
    result_window = Tk()
    result_window.title("Search Results")

    for filename, sim in results:
        Label(result_window, text=f"{filename}: {sim:.2f} similarity").pack()

    result_window.mainloop()


if __name__ == "__main__":
    select_folder()
    main_gui()
