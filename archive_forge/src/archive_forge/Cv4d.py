# Import necessary modules
import tkinter as tk  # Import tkinter for creating GUI
from tkinter import (
    filedialog,
    Label,
    Button,
    Frame,
    messagebox,
    ttk,
)  # Import specific tkinter widgets
import os  # Import os module for file and directory operations
import hashlib  # Import hashlib for generating file hashes
from pathlib import Path  # Import Path for file path handling
import ast  # Import ast module for parsing Python code
import json  # Import json module for JSON serialization and deserialization
import sys  # Import sys module for system-specific parameters and functions
import importlib.util  # Import importlib.util for importing modules dynamically
import inspect  # Import inspect module for introspecting live objects
import logging  # Import logging module for logging errors and messages


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


if __name__ == "__main__":
    root = tk.Tk()
    app = FileProcessor(root)
    root.mainloop()
