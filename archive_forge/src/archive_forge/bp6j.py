import os
import hashlib
import ast
import json
import logging
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from typing import Optional, Type, Tuple, List, Dict, Callable
import tkinter as tk
from tkinter import filedialog, messagebox

# Configure logging with a detailed format specification
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SQLiteConnectionManager:
    """
    Manages the lifecycle of SQLite database connections, ensuring robust, reusable, and efficient database connectivity.
    This class encapsulates connection handling and transaction management, serving as a foundational component for database interactions.
    It adheres to the context management protocol to facilitate safe use of resources.
    """

    def __init__(self, database_path: str):
        """
        Initializes an SQLiteConnectionManager instance with a specified path to the SQLite database.
        Parameters:
        - database_path (str): The filesystem path to the SQLite database file.
        """
        self.database_path: str = database_path
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def __enter__(self) -> "SQLiteConnectionManager":
        """
        Establishes a database connection upon entering the runtime context, preparing the manager for database operations.
        Returns:
        - SQLiteConnectionManager: The instance with an active database connection.
        """
        self.connection = sqlite3.connect(self.database_path)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Type[BaseException]],
    ) -> None:
        """
        Terminates the database connection upon exiting the runtime context, ensuring clean disconnection and handling exceptions gracefully.
        Parameters:
        - exc_type (Optional[Type[BaseException]]): The type of the exception, if any.
        - exc_val (Optional[BaseException]): The exception instance, if any.
        - exc_tb (Optional[Type[BaseException]]): The traceback object, if any.
        """
        if self.connection:
            if exc_type is not None:
                self.connection.rollback()
            else:
                self.connection.commit()
            self.connection.close()

    def commit_changes(self) -> None:
        """
        Commits the current transaction to the database, ensuring that all changes made during the transaction are persisted.
        This method is critical for maintaining data integrity and consistency within the database.
        """
        if self.connection:
            self.connection.commit()


class EmbeddingDatabaseManager:
    """
    Manages the database operations specifically for embeddings, including initialization, insertion, retrieval, and closure.
    """

    def __init__(self, connection_manager: SQLiteConnectionManager):
        """
        Initializes the EmbeddingDatabaseManager with a connection manager to handle database operations.
        Parameters:
        - connection_manager (SQLiteConnectionManager): The connection manager to handle database interactions.
        """
        self.connection_manager = connection_manager
        self.initialize_database()

    def initialize_database(self) -> None:
        """
        Initialize the database by creating the necessary table if it does not exist.
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                file_hash TEXT UNIQUE,
                imports TEXT,
                function TEXT,
                docstring TEXT,
                embedding BLOB
            )
        """
        self.connection_manager.cursor.execute(create_table_query)
        self.connection_manager.commit_changes()

    def insert_embedding(
        self,
        file_hash: str,
        imports: str,
        function: str,
        docstring: str,
        embedding: bytes,
    ) -> None:
        """
        Insert an embedding into the database if the file has not been processed before.
        This method first checks if the file hash already exists in the database to avoid duplicate entries.
        Parameters:
        - file_hash: str - The hash of the file.
        - imports: str - The imports used in the file.
        - function: str - The function code.
        - docstring: str - The docstring of the function.
        - embedding: bytes - The serialized embedding.
        """
        if not self.is_file_processed(file_hash):
            try:
                self.connection_manager.cursor.execute(
                    """
                    INSERT INTO embeddings (file_hash, imports, function, docstring, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (file_hash, imports, function, docstring, embedding),
                )
                logging.info(f"Embedding inserted for file hash: {file_hash}")
            except sqlite3.IntegrityError as e:
                logging.error(
                    f"Error inserting embedding: {e} for file hash: {file_hash}"
                )
        else:
            logging.info(f"Skipped insertion; already processed file hash: {file_hash}")

    def is_file_processed(self, file_hash: str) -> bool:
        """
        Check if a file has already been processed by querying its hash.
        Parameters:
        - file_hash: str - The hash of the file to check.
        :return: bool - True if the file has been processed, False otherwise.
        """
        check_query = "SELECT 1 FROM embeddings WHERE file_hash = ?"
        try:
            self.connection_manager.cursor.execute(check_query, (file_hash,))
            return self.connection_manager.cursor.fetchone() is not None
        except sqlite3.DatabaseError as e:
            logging.error(f"Error checking processed file: {e}")
            raise

    def fetch_embeddings(self, search_term: str) -> List[Tuple[str, str, str, bytes]]:
        """
        Retrieve embeddings that match the search term either in function or docstring.
        Parameters:
        - search_term: str - The term to search for in function names or docstrings.
        :return: List[Tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_query = """
            SELECT imports, function, docstring, embedding 
            FROM embeddings 
            WHERE function LIKE ? OR docstring LIKE ? COLLATE NOCASE
        """
        try:
            self.connection_manager.cursor.execute(
                retrieve_query, ("%" + search_term + "%", "%" + search_term + "%")
            )
            return self.connection_manager.cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving embeddings: {e}")
            raise

    def fetch_all_embeddings(self) -> List[Tuple[str, str, str, bytes]]:
        """
        Retrieve all embeddings from the database using pagination to handle large datasets efficiently.
        :return: List[Tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_all_query = (
            "SELECT imports, function, docstring, embedding FROM embeddings"
        )
        try:
            self.connection_manager.cursor.execute(retrieve_all_query)
            results = []
            while True:
                page = self.connection_manager.cursor.fetchmany(
                    1000
                )  # Fetch results in pages of 1000
                if not page:
                    break
                results.extend(page)
            return results
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving all embeddings: {e}")
            raise


# Define FileProcessor class
class FileProcessor:
    """
    A class for processing Python files and organizing them into separate components.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the FileProcessor instance.

        Args:
            root (tk.Tk): The root window of the GUI application.
        """
        self.root = root  # Store the root window reference
        self.setup_ui()  # Set up the user interface
        self.processed_files = (
            self.load_processed_files()
        )  # Load previously processed files

    def setup_ui(self) -> None:
        """
        Set up the user interface for the FileProcessor.
        """
        self.root.title("Python File Organizer")  # Set the title of the root window
        self.frame = Frame(self.root)  # Create a frame to hold the widgets
        self.frame.pack(padx=10, pady=10)  # Pack the frame with padding

        # Create and pack the input folder selection button
        self.input_button = Button(
            self.frame, text="Select Input Folder", command=self.select_input_folder
        )
        self.input_button.pack(fill=tk.X)

        # Create and pack the output folder selection button
        self.output_button = Button(
            self.frame, text="Select Output Folder", command=self.select_output_folder
        )
        self.output_button.pack(fill=tk.X)

        # Create and pack the process files button
        self.process_button = Button(
            self.frame, text="Process Files", command=self.process_files
        )
        self.process_button.pack(fill=tk.X)

        # Create and pack the progress bar
        self.progress = ttk.Progressbar(
            self.frame, orient=tk.HORIZONTAL, length=300, mode="determinate"
        )
        self.progress.pack(fill=tk.X, pady=(10, 0))

        # Create and pack the status label
        self.status_label = Label(self.frame, text="", wraplength=400, justify="left")
        self.status_label.pack(fill=tk.X, pady=(10, 0))

    def select_input_folder(self) -> None:
        """
        Open a file dialog to select the input folder.
        """
        self.input_folder = (
            filedialog.askdirectory()
        )  # Open a directory selection dialog
        if self.input_folder:
            # Update the status label with the selected input folder
            self.status_label.config(
                text=f"Selected Input Folder:\n{self.input_folder}"
            )

    def select_output_folder(self) -> None:
        """
        Open a file dialog to select the output folder.
        """
        self.output_folder = (
            filedialog.askdirectory()
        )  # Open a directory selection dialog
        if self.output_folder:
            # Update the status label with the selected output folder
            self.status_label.config(
                text=f"Selected Output Folder:\n{self.output_folder}"
            )

    def process_files(self) -> None:
        """
        Process the Python files in the selected input folder and organize them in the output folder.
        """
        # Check if both input and output folders are selected
        if not hasattr(self, "input_folder") or not hasattr(self, "output_folder"):
            messagebox.showerror(
                "Error", "Please select both input and output folders."
            )
            return

        # Find all Python files in the input folder and its subfolders
        python_files = list(Path(self.input_folder).rglob("*.py"))
        total_files = len(python_files)
        self.progress["maximum"] = total_files

        # Process each Python file
        for index, file_path in enumerate(python_files, start=1):
            # Update the status label with the current file being processed
            self.status_label.config(
                text=f"Processing:\n{file_path.name}\n({index}/{total_files})"
            )
            self.progress["value"] = index
            self.root.update_idletasks()

            file_hash = self.hash_file(file_path)  # Generate a hash of the file
            if file_hash in self.processed_files:
                continue  # Skip processing if the file has been processed before

            try:
                if file_path.is_dir():
                    # Log an error if the file path is a directory
                    logging.error(f"Skipping directory {file_path} with .py extension.")
                    continue

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()  # Read the file content
                tree = ast.parse(content)  # Parse the Python code into an AST

                # Extract classes, functions, and documentation from the AST
                classes, functions, docs = self.extract_components(tree)
                imports = self.extract_imports(content)  # Extract import statements
                # Save the extracted components to separate files
                self.save_components(classes, functions, docs, imports, file_path.stem)

                # Update the processed files dictionary
                self.processed_files[file_hash] = {"path": str(file_path)}
                self.save_processed_files()  # Save the processed files dictionary
            except Exception as e:
                # Log an error if processing fails for a file
                logging.error(f"Failed to process {file_path.name}: {str(e)}")
                continue

        self.status_label.config(text="Processing complete.")
        self.progress["value"] = 0

    def hash_file(self, file_path: Path) -> str:
        """
        Generate a SHA-256 hash of the given file. This function ensures that the path provided is a file and not a directory.
        If the path is a directory, it logs an error and continues processing without interruption.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The SHA-256 hash of the file, or an empty string if the path is a directory.
        """
        if not file_path.is_file():
            # Log an error if the path is not a file
            logging.error(f"Expected a file but received a directory: {file_path}")
            return ""

        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as file:
                while True:
                    buf = file.read(65536)  # Read in 64kb chunks to handle large files
                    if not buf:
                        break
                    hasher.update(buf)
        except Exception as e:
            logging.error(f"Failed to read and hash the file {file_path}: {str(e)}")
            return ""
        return hasher.hexdigest()

    def extract_components(self, tree: ast.AST):
        """
        Extract classes, functions, and documentation from the given AST.

        Args:
            tree (ast.AST): The AST representing the Python code.

        Returns:
            tuple: A tuple containing lists of classes, functions, and documentation.
        """
        classes = []
        functions = []
        docs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append((node.name, ast.unparse(node)))
            elif isinstance(node, ast.FunctionDef):
                functions.append((node.name, ast.unparse(node)))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                docs.append((node.value.s,))

        return classes, functions, docs

    def extract_imports(self, content: str) -> str:
        """
        Extract all import statements from the given content.

        Args:
            content (str): The Python code content.

        Returns:
            str: The extracted import statements.
        """
        imports = ""
        for line in content.splitlines():
            if line.startswith("import") or line.startswith("from"):
                imports += line + "\n"
        return imports

    def save_components(self, classes, functions, docs, imports, base_name):
        """
        Save the extracted components to separate files in the output folder.

        Args:
            classes (list): A list of extracted classes.
            functions (list): A list of extracted functions.
            docs (list): A list of extracted documentation strings.
            imports (str): The extracted import statements.
            base_name (str): The base name for the output files.
        """
        class_folder = Path(self.output_folder) / "classes"
        function_folder = Path(self.output_folder) / "functions"
        doc_folder = Path(self.output_folder) / "docs"

        # Create the output folders if they don't exist
        class_folder.mkdir(parents=True, exist_ok=True)
        function_folder.mkdir(parents=True, exist_ok=True)
        doc_folder.mkdir(parents=True, exist_ok=True)

        # Save each class to a separate file
        for name, content in classes:
            with open(class_folder / f"class_{name}.py", "w", encoding="utf-8") as file:
                file.write(imports + content)

        # Save each function to a separate file
        for name, content in functions:
            with open(
                function_folder / f"func_{name}.py", "w", encoding="utf-8"
            ) as file:
                file.write(imports + content)

        # Save each documentation string to a separate file
        for (content,) in docs:
            with open(
                doc_folder / f"{base_name}_documentation.py", "w", encoding="utf-8"
            ) as file:
                file.write(imports + content)

    def load_processed_files(self) -> dict:
        """
        Load the previously processed files from a JSON file.

        Returns:
            dict: A dictionary containing the processed files.
        """
        try:
            with open("processed_files.json", "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save_processed_files(self) -> None:
        """
        Save the processed files to a JSON file.
        """
        with open("processed_files.json", "w", encoding="utf-8") as file:
            json.dump(self.processed_files, file, indent=4)


class EmbeddingSearchGUI:
    """
    Manages the application's main window and interactions, encapsulating all related functionalities.
    """

    def __init__(
        self,
        root: tk.Tk,
        embedding_database_manager: EmbeddingDatabaseManager,
        embedding_model: SentenceTransformer,
        file_manager: FileManager,
    ):
        """
        Initializes the GUI with all necessary components and managers.
        :param root: tk.Tk - The root window of the application.
        :param embedding_database_manager: EmbeddingDatabaseManager - The manager for handling embedding database operations.
        :param embedding_model: SentenceTransformer - The model used for generating embeddings.
        :param file_manager: FileManager - The manager for handling file-related operations.
        """
        self.root = root
        self.embedding_database_manager = embedding_database_manager
        self.embedding_model = embedding_model
        self.file_manager = file_manager
        self._initialize_user_interface()

    def _initialize_user_interface(self) -> None:
        """
        Configures the user interface, creating and organizing widgets.
        """
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.input_button = tk.Button(
            self.frame,
            text="Select Python Files Folder",
            command=self._handle_folder_selection,
        )
        self.input_button.pack(fill=tk.X)

        self.search_frame = tk.Frame(self.frame)
        self.search_frame.pack(fill=tk.X)

        self.search_entry = tk.Entry(self.search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.insert(0, "Enter search term...")

        self.search_button = tk.Button(
            self.search_frame,
            text="Search Embeddings",
            command=self._execute_embedding_search,
        )
        self.search_button.pack(side=tk.LEFT)

        self.results_text = tk.Text(self.frame, height=15, width=80)
        self.results_text.pack(fill=tk.X)

    def _handle_folder_selection(self) -> None:
        """
        Manages folder selection and initiates processing of files within the selected folder.
        """
        folder_path = filedialog.askdirectory()
        if folder_path:
            total_files, processed_files = self.file_manager.process_folder(
                folder_path,
                lambda file_path: self.file_manager.process_file(
                    file_path, self.embedding_model, self.embedding_database_manager
                ),
            )
            self.embedding_database_manager.connection_manager.commit_changes()
            tk.messagebox.showinfo(
                "Processing Complete",
                f"All files have been processed. {processed_files}/{total_files} files processed.",
            )
        else:
            tk.messagebox.showinfo(
                "No folder selected", "Please select a valid folder."
            )

    def _execute_embedding_search(self) -> None:
        """
        Executes a search for embeddings based on the user's input and displays the results.
        """
        search_term = self.search_entry.get()
        results = self.embedding_database_manager.fetch_embeddings(search_term)
        self._present_search_results(results)

    def _present_search_results(
        self, results: List[Tuple[str, str, str, bytes]]
    ) -> None:
        """
        Presents search results in the text widget.

        :param results: List[Tuple[str, str, str, bytes]] - A list of tuples containing the imports, function, docstring, and embedding for each search result.
        """
        self.results_text.delete("1.0", tk.END)
        for imports, function, docstring, _ in results:
            result_text = f"Imports:\n{imports}\n\nFunction:\n{function}\n\nDocstring:\n{docstring}\n\n---\n\n"
            self.results_text.insert(tk.END, result_text)
        logging.info("Search results displayed.")


def main():
    """
    The main entry point of the application.
    """
    root = tk.Tk()
    root.title("Embedding Search")

    with SQLiteConnectionManager("embeddings.db") as connection_manager:
        embedding_database_manager = EmbeddingDatabaseManager(connection_manager)
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        processed_files_path = "processed_files.json"
        embedded_files_path = "embedded_files.json"
        file_manager = FileManager(processed_files_path, embedded_files_path)

        gui = EmbeddingSearchGUI(
            root, embedding_database_manager, embedding_model, file_manager
        )
        root.mainloop()


if __name__ == "__main__":
    main()
