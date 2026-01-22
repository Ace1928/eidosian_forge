import os
import sqlite3
import hashlib
from sentence_transformers import SentenceTransformer
from tkinter import filedialog, Tk, Button, Label, Entry, messagebox
import textract
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Sentence Transformer Model
model = SentenceTransformer("all-mpnet-base-v2")

# Database setup
conn = sqlite3.connect("embeddings.db")
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, embeddings BLOB, filename TEXT)"""
)
conn.commit()


def hash_file(filepath):
    """Create a hash for a file to check if it has been processed."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_text_from_file(filepath):
    """Extract text from various file formats."""
    try:
        return textract.process(filepath).decode("utf-8")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def store_embeddings(text, file_hash, filename):
    """Generate embeddings and store them in the database."""
    embeddings = model.encode([text], show_progress_bar=True)
    c.execute(
        "INSERT OR IGNORE INTO documents (hash, embeddings, filename) VALUES (?, ?, ?)",
        (file_hash, embeddings.tobytes(), filename),
    )
    conn.commit()


def process_folder(folder_path):
    """Process all files in the selected folder."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_hash = hash_file(filepath)
            c.execute("SELECT hash FROM documents WHERE hash = ?", (file_hash,))
            if c.fetchone() is None:
                if text := extract_text_from_file(filepath):
                    store_embeddings(text, file_hash, file)
                else:
                    print(f"Failed to process {filepath}")


def select_folder():
    """GUI to select a folder for processing."""
    root = Tk()
    root.title("Document Processor")

    Label(root, text="Select a folder to process:").pack()
    Button(root, text="Select Folder", command=lambda: process_and_close(root)).pack()
    root.mainloop()


def process_and_close(root):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        process_folder(folder_selected)
        messagebox.showinfo("Process Complete", "All files have been processed.")
    root.destroy()


def load_embeddings():
    """Load embeddings from the database for analysis."""
    c.execute("SELECT embeddings, filename FROM documents")
    data = c.fetchall()
    embeddings = [np.frombuffer(d[0], dtype=np.float32) for d in data]
    filenames = [d[1] for d in data]
    return embeddings, filenames


def search_embeddings(query, num_results=5):
    """Search for similar documents based on cosine similarity."""
    query_embedding = model.encode([query])[0]
    embeddings, filenames = load_embeddings()
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    return [(filenames[i], similarities[i]) for i in top_indices]


def main_gui():
    root = Tk()
    root.title("Search Documents")

    Label(root, text="Enter search query:").pack()
    query_input = Entry(root, width=50)
    query_input.pack()

    Button(
        root, text="Search", command=lambda: show_results(query_input.get(), root)
    ).pack()

    root.mainloop()


def show_results(query, root):
    results = search_embeddings(query)
    result_window = Tk()
    result_window.title("Search Results")

    for filename, sim in results:
        Label(result_window, text=f"{filename}: {sim:.2f} similarity").pack()

    result_window.mainloop()


if __name__ == "__main__":
    select_folder()
    main_gui()
