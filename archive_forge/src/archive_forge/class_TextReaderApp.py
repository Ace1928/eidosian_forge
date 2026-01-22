import tkinter as tk
from tkinter import filedialog, scrolledtext
import requests
from bs4 import BeautifulSoup
import json
class TextReaderApp:

    def __init__(self, root):
        self.root = root
        root.title('Text Reader')
        self.url_entry = tk.Entry(root, width=50)
        self.url_entry.pack()
        self.read_button = tk.Button(root, text='Read Text', command=self.read_text)
        self.read_button.pack()
        self.load_json_button = tk.Button(root, text='Load JSON', command=self.load_json_action)
        self.load_json_button.pack()
        self.json_section_entry = tk.Entry(root, width=50)
        self.json_section_entry.pack()
        self.text_area = scrolledtext.ScrolledText(root, height=15, width=80)
        self.text_area.pack()

    def read_text(self):
        url = self.url_entry.get()
        if url:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                self.text_area.insert(tk.END, text)
            except Exception as e:
                self.text_area.insert(tk.END, f'Error: {e}')

    def load_json_action(self):
        file_path = filedialog.askopenfilename(filetypes=[('JSON files', '*.json')])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    self.text_area.insert(tk.END, json.dumps(json_data, indent=4))
            except Exception as e:
                self.text_area.insert(tk.END, f'Error loading JSON: {e}')