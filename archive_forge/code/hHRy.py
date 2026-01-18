import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import math


class PyramidDecoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pyramid Decoder")
        self.setup_ui()

    def setup_ui(self):
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.load_button = Button(self.frame, text="Load File", command=self.load_file)
        self.load_button.pack(fill=tk.X)

        self.decode_button = Button(
            self.frame,
            text="Decode Message",
            command=self.decode_message,
            state=tk.DISABLED,
        )
        self.decode_button.pack(fill=tk.X)

        self.message_label = Label(self.frame, text="Decoded message will appear here.")
        self.message_label.pack(fill=tk.X)

        self.canvas = tk.Canvas(self.root, width=400, height=300)
        self.canvas.pack(side=tk.RIGHT)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.file_path = file_path
            self.decode_button.config(state=tk.NORMAL)
            self.message_label.config(
                text="File loaded. Press 'Decode Message' to decode."
            )

    def decode_message(self):
        number_word_map = {}
        with open(self.file_path, "r") as file:
            for line in file:
                parts = line.split()
                number = int(parts[0])
                word = " ".join(parts[1:])
                number_word_map[number] = word

        max_number = max(number_word_map.keys())
        last_numbers = []
        current_number = 1
        level = 1
        self.canvas.delete("all")
        y = 20

        while current_number <= max_number:
            last_number = current_number + level - 1
            if last_number <= max_number:
                last_numbers.append(last_number)
                self.canvas.create_text(
                    200,
                    y,
                    text=f"Level {level}: {last_number} = {number_word_map.get(last_number, '')}",
                )
                y += 20
            current_number = last_number + 1
            level += 1

        words = [number_word_map[num] for num in last_numbers if num in number_word_map]
        message = " ".join(words)
        self.message_label.config(text=f"Decoded message: {message}")


# Example usage
root = tk.Tk()
app = PyramidDecoderApp(root)
root.mainloop()
