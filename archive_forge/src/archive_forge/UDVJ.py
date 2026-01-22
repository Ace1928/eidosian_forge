import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox, ttk
import os
import hashlib
from pathlib import Path
import ast
import json


class FileProcessor:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.processed_files = self.load_processed_files()

    def setup_ui(self):
        self.root.title("Python File Organizer")
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.input_button = Button(
            self.frame, text="Select Input Folder", command=self.select_input_folder
        )
        self.input_button.pack(fill=tk.X)

        self.output_button = Button(
            self.frame, text="Select Output Folder", command=self.select_output_folder
        )
        self.output_button.pack(fill=tk.X)

        self.process_button = Button(
            self.frame, text="Process Files", command=self.process_files
        )
        self.process_button.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            self.frame, orient=tk.HORIZONTAL, length=100, mode="determinate"
        )
        self.progress.pack(fill=tk.X)

        self.status_label = Label(self.frame, text="", wraplength=300)
        self.status_label.pack(fill=tk.X)

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            self.status_label.config(text=f"Selected Input Folder: {self.input_folder}")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        if self.output_folder:
            self.status_label.config(
                text=f"Selected Output Folder: {self.output_folder}"
            )

    def process_files(self):
        if not hasattr(self, "input_folder") or not hasattr(self, "output_folder"):
            messagebox.showerror(
                "Error", "Please select both input and output folders."
            )
            return

        python_files = [f for f in Path(self.input_folder).rglob("*.py")]
        total_files = len(python_files)
        self.progress["maximum"] = total_files

        for index, file_path in enumerate(python_files, start=1):
            self.status_label.config(
                text=f"Processing {file_path.name} ({index}/{total_files})"
            )
            self.progress["value"] = index
            self.root.update_idletasks()

            file_hash = self.hash_file(file_path)
            if file_hash in self.processed_files:
                continue  # Skip processing if file has been processed before

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                tree = ast.parse(content)

                classes, functions, docs = self.extract_components(tree)
                self.save_components(classes, functions, docs, file_path.stem)

                self.processed_files[file_hash] = {"path": str(file_path)}
                self.save_processed_files()
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to process {file_path.name}: {str(e)}"
                )
                continue

        self.status_label.config(text="Processing complete.")
        self.progress["value"] = 0

    def hash_file(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as file:
            buf = file.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def extract_components(self, tree):
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

    def save_components(self, classes, functions, docs, base_name):
        class_folder = Path(self.output_folder) / "classes"
        function_folder = Path(self.output_folder) / "functions"
        doc_folder = Path(self.output_folder) / "docs"

        class_folder.mkdir(parents=True, exist_ok=True)
        function_folder.mkdir(parents=True, exist_ok=True)
        doc_folder.mkdir(parents=True, exist_ok=True)

        for name, content in classes:
            with open(class_folder / f"class_{name}.py", "w", encoding="utf-8") as file:
                file.write(content)

        for name, content in functions:
            with open(
                function_folder / f"func_{name}.py", "w", encoding="utf-8"
            ) as file:
                file.write(content)

        for (content,) in docs:
            with open(
                doc_folder / f"{base_name}_documentation.py", "w", encoding="utf-8"
            ) as file:
                file.write(content)

    def load_processed_files(self):
        try:
            with open("processed_files.json", "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save_processed_files(self):
        with open("processed_files.json", "w", encoding="utf-8") as file:
            json.dump(self.processed_files, file, indent=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = FileProcessor(root)
    root.mainloop()
