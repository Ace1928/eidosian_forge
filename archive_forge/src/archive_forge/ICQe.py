import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas
import math


class PyramidDecoderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pyramid Decoder")
        self.setup_ui()

    def setup_ui(self) -> None:
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

        self.canvas = Canvas(self.root, width=600, height=800)
        self.canvas.pack(side=tk.RIGHT)
        self.draw_pyramid()

    def draw_pyramid(self) -> None:
        y = 20
        x_center = 300
        number = 1
        level = 1
        while y < 800:  # Limiting the pyramid drawing to the canvas height
            x = x_center - 10 * level
            for i in range(level):
                self.canvas.create_text(x, y, text=str(number), font=("Helvetica", 10))
                x += 20
                number += 1
            self.canvas.create_text(
                x_center,
                y + 15,
                text=f"Row {level} ends with {number - 1}",
                font=("Helvetica", 10, "bold"),
            )
            y += 30
            level += 1

    def load_file(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.file_path = file_path
            self.decode_button.config(state=tk.NORMAL)
            self.message_label.config(
                text="File loaded. Press 'Decode Message' to decode."
            )

    def decode_message(self) -> None:
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
        y = 20

        while current_number <= max_number:
            last_number = current_number + level - 1
            if last_number <= max_number:
                last_numbers.append(last_number)
            current_number = last_number + 1
            level += 1

        words = [number_word_map[num] for num in last_numbers if num in number_word_map]
        message = " ".join(words)
        self.message_label.config(text=f"Decoded message: {message}")


# Example usage
root = tk.Tk()
app = PyramidDecoderApp(root)
root.mainloop()
