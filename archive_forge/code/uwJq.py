import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame, Canvas
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PyramidNumbersApp:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the PyramidNumbersApp with the given Tk root.

        Args:
            root (tk.Tk): The root window for the application.
        """
        self.root = root
        self.root.title("Pyramid Numbers Decoder")
        self.setup_ui()

    def setup_ui(self) -> None:
        """
        Set up the user interface for the Pyramid Numbers application.
        """
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.load_button = Button(self.frame, text="Load File", command=self.load_file)
        self.load_button.pack(fill=tk.X)

        self.decode_button = Button(
            self.frame,
            text="Decode Pyramid Numbers",
            command=self.decode_pyramid_numbers,
            state=tk.DISABLED,
        )
        self.decode_button.pack(fill=tk.X)

        self.message_label = Label(self.frame, text="Decoded message will appear here.")
        self.message_label.pack(fill=tk.X)

        self.canvas = Canvas(self.root, width=600, height=800)
        self.canvas.pack(side=tk.RIGHT)

    def load_file(self) -> None:
        """
        Prompt the user to select a file and enable the decode button if a file is selected.
        """
        logging.info("Prompting user to select a file.")
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.file_path = file_path
            self.decode_button.config(state=tk.NORMAL)
            self.message_label.config(
                text="File loaded. Press 'Decode Pyramid Numbers' to decode."
            )
            logging.info(f"File loaded: {file_path}")
        else:
            messagebox.showinfo("No file selected", "Please select a valid text file.")
            logging.warning("No file selected by the user.")

    def decode_pyramid_numbers(self) -> None:
        """
        Decode the pyramid numbers from the loaded file and display the corresponding message.
        """
        logging.info("Decoding pyramid numbers.")
        number_word_map = self.read_file(self.file_path)
        if number_word_map:
            max_number = max(number_word_map.keys())
            pyramid_nums = self.calculate_pyramid_numbers(max_number)
            words = [
                number_word_map[num] for num in pyramid_nums if num in number_word_map
            ]
            message = " ".join(words)
            self.display_message(message)
            self.display_pyramid_numbers_with_words(pyramid_nums, number_word_map)
            logging.info("Pyramid numbers decoded and displayed.")

    def read_file(self, file_path: str) -> Dict[int, str]:
        """
        Read the file at the given path and return a dictionary mapping numbers to words.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            Dict[int, str]: A dictionary mapping numbers to words.
        """
        number_word_map: Dict[int, str] = {}
        try:
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:
                        number = int(parts[0])
                        word = " ".join(parts[1:])
                        number_word_map[number] = word
            logging.info(f"File read successfully: {file_path}")
        except Exception as e:
            messagebox.showerror(
                "File Error", f"An error occurred while reading the file: {str(e)}"
            )
            logging.error(f"Error reading file: {file_path}, {str(e)}")
        return number_word_map

    def calculate_pyramid_numbers(self, max_number: int) -> List[int]:
        """
        Calculate the pyramid numbers up to the maximum number.

        Args:
            max_number (int): The maximum number to calculate pyramid numbers for.

        Returns:
            List[int]: A list of pyramid numbers up to the maximum number.
        """
        pyramid_numbers: List[int] = []
        current_number = 1
        level = 1
        while current_number <= max_number:
            last_number = current_number + level - 1
            pyramid_numbers.append(last_number)
            current_number = last_number + 1
            level += 1
        logging.info("Pyramid numbers calculated.")
        return pyramid_numbers

    def display_message(self, message: str) -> None:
        """
        Display the given message in the message label.

        Args:
            message (str): The message to display.
        """
        self.message_label.config(text=f"Decoded message: {message}")
        logging.info(f"Message displayed: {message}")

    def display_pyramid_numbers_with_words(
        self, pyramid_nums: List[int], number_word_map: Dict[int, str]
    ) -> None:
        """
        Display the pyramid numbers along with their corresponding words on the canvas.

        Args:
            pyramid_nums (List[int]): List of pyramid numbers.
            number_word_map (Dict[int, str]): Mapping of numbers to words.
        """
        y = 20
        x_center = 300
        for num in pyramid_nums:
            word = number_word_map.get(num, "")
            self.canvas.create_text(
                x_center, y, text=f"{num}: {word}", font=("Helvetica", 10)
            )
            y += 30
        logging.info("Pyramid numbers with words displayed on canvas.")


# Example usage
root = tk.Tk()
app = PyramidNumbersApp(root)
root.mainloop()
