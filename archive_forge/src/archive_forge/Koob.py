import sys
import logging
from typing import List, Tuple, Generator
from functools import wraps
from time import perf_counter
from contextlib import contextmanager
from memory_profiler import profile
import multiprocessing
import argparse
import itertools
import functools
import psutil

# Import necessary modules for GUI development
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Importing necessary modules for Unicode character handling, color formatting, performance profiling, resource management, parallel processing, and user interaction
from typing import List, Generator


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time:.5f} seconds")
        return result

    return wrapper


def log_memory_usage_decorator(func):
    """
    Decorator to log the memory usage before and after executing a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        before = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        after = psutil.Process().memory_info().rss
        logging.info(
            f"Memory usage of {func.__name__}: {before} bytes -> {after} bytes"
        )
        return result

    return wrapper


# Define the function to programmatically generate characters using memory-efficient generators
@log_execution_time
@log_memory_usage_decorator
def generate_utf_characters(
    custom_ranges: List[Tuple[int, int]] = None
) -> Generator[str, None, None]:
    """
    Generates a comprehensive list of UTF block, pipe, shape, and other related characters by systematically iterating through Unicode code points using memory-efficient generators.

    Args:
        custom_ranges (List[Tuple[int, int]], optional): Custom Unicode ranges to include in the generation process. Defaults to None.

    Returns:
        Generator[str, None, None]: A generator yielding unique UTF characters including block, pipe, shape, and other related characters.
    """
    default_ranges: List[Tuple[int, int]] = [
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x2700, 0x27BF),  # Dingbats
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    ]

    if custom_ranges:
        ranges = custom_ranges
    else:
        ranges = default_ranges

    unique_chars = set()

    for start, end in ranges:
        for code_point in range(start, end + 1):
            char = chr(code_point)
            if char not in unique_chars:
                unique_chars.add(char)
                yield char


class UnicodeCharacterGeneratorGUI(tk.Tk):
    """
    A graphical user interface for generating and displaying Unicode characters.
    """

    def __init__(self):
        super().__init__()

        self.title("Unicode Character Generator")
        self.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets for the GUI.
        """
        # Create the main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options")
        options_frame.pack(fill="x", padx=5, pady=5)

        # Create the font type label and combobox
        font_type_label = ttk.Label(options_frame, text="Font Type:")
        font_type_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.font_type_var = tk.StringVar()
        font_type_combobox = ttk.Combobox(
            options_frame, textvariable=self.font_type_var, values=font.families()
        )
        font_type_combobox.set("Arial")
        font_type_combobox.grid(row=0, column=1, padx=5, pady=5)

        # Create the character size label and spinbox
        character_size_label = ttk.Label(options_frame, text="Character Size:")
        character_size_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.character_size_var = tk.IntVar()
        character_size_spinbox = ttk.Spinbox(
            options_frame, from_=10, to=100, textvariable=self.character_size_var
        )
        character_size_spinbox.set(20)
        character_size_spinbox.grid(row=1, column=1, padx=5, pady=5)

        # Create the custom Unicode ranges label and entry
        custom_ranges_label = ttk.Label(options_frame, text="Custom Unicode Ranges:")
        custom_ranges_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.custom_ranges_var = tk.StringVar()
        custom_ranges_entry = ttk.Entry(
            options_frame, textvariable=self.custom_ranges_var
        )
        custom_ranges_entry.grid(row=2, column=1, padx=5, pady=5)

        # Create the output destination label and radiobuttons
        output_destination_label = ttk.Label(options_frame, text="Output Destination:")
        output_destination_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.output_destination_var = tk.StringVar()
        console_radiobutton = ttk.Radiobutton(
            options_frame,
            text="Console",
            variable=self.output_destination_var,
            value="console",
        )
        console_radiobutton.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        file_radiobutton = ttk.Radiobutton(
            options_frame,
            text="File",
            variable=self.output_destination_var,
            value="file",
        )
        file_radiobutton.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.output_destination_var.set("console")

        # Create the output file label, entry, and browse button
        output_file_label = ttk.Label(options_frame, text="Output File:")
        output_file_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.output_file_var = tk.StringVar()
        output_file_entry = ttk.Entry(options_frame, textvariable=self.output_file_var)
        output_file_entry.grid(row=5, column=1, padx=5, pady=5)
        browse_button = ttk.Button(
            options_frame, text="Browse", command=self.browse_output_file
        )
        browse_button.grid(row=5, column=2, padx=5, pady=5)

        # Create the batch size label and spinbox
        batch_size_label = ttk.Label(options_frame, text="Batch Size:")
        batch_size_label.grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.batch_size_var = tk.IntVar()
        batch_size_spinbox = ttk.Spinbox(
            options_frame, from_=100, to=10000, textvariable=self.batch_size_var
        )
        batch_size_spinbox.set(1000)
        batch_size_spinbox.grid(row=6, column=1, padx=5, pady=5)

        # Create the generate button
        generate_button = ttk.Button(
            main_frame, text="Generate", command=self.generate_characters
        )
        generate_button.pack(pady=10)

        # Create the character display frame
        display_frame = ttk.LabelFrame(main_frame, text="Generated Characters")
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create the character text widget
        self.character_text = tk.Text(display_frame, wrap="char", state="normal")
        self.character_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Configure the text widget to be read-only
        self.character_text.configure(state="disabled")

    def browse_output_file(self):
        """
        Opens a file dialog to select the output file.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if file_path:
            self.output_file_var.set(file_path)

    def generate_characters(self):
        """
        Generates Unicode characters based on the selected options and displays them in the GUI or saves them to a file.
        """
        # Get the selected options
        font_type = self.font_type_var.get()
        character_size = self.character_size_var.get()
        custom_ranges_str = self.custom_ranges_var.get()
        output_destination = self.output_destination_var.get()
        output_file = self.output_file_var.get()
        batch_size = self.batch_size_var.get()

        # Configure the character text widget font
        self.character_text.configure(state="normal")
        self.character_text.configure(font=(font_type, character_size))

        # Process custom Unicode ranges
        custom_ranges = []
        for range_str in custom_ranges_str.split(","):
            range_str = range_str.strip()
            if range_str:
                start, end = map(lambda x: int(x, 16), range_str.split("-"))
                custom_ranges.append((start, end))

        # Generate UTF characters based on the provided ranges
        characters = generate_utf_characters(custom_ranges)

        # Generate the color spectrum
        colors = self.generate_color_spectrum()

        # Print or save the characters based on the output option
        if output_destination == "console":
            # Clear the text widget
            self.character_text.delete("1.0", "end")

            # Use parallel processing to speed up character printing
            with multiprocessing.Pool() as pool:
                batches = itertools.islice(characters, batch_size)
                for batch in batches:
                    # Print characters in the text widget
                    for char, color in zip(batch, itertools.cycle(colors)):
                        self.character_text.insert("end", f"{char}", "color")
                        self.character_text.tag_config("color", foreground=color)
                    self.character_text.insert("end", "\n")
        else:
            # Save characters to a file
            with open(output_file, "w", encoding="utf-8") as file:
                for char in characters:
                    file.write(f"{char}\n")
            messagebox.showinfo("File Saved", f"Characters saved to {output_file}")

    def generate_color_spectrum(self, steps: int = 256) -> List[str]:
        """
        Generates a color spectrum.

        Args:
            steps (int): The number of colors to generate.

        Returns:
            List[str]: A list of hexadecimal color codes.
        """
        spectrum = []
        # Define key colors in the spectrum
        colors = [
            "#000000",
            "#808080",  # Black to gray
            "#FFFFFF",  # White
            "#FF0000",
            "#FFA500",  # Red to orange
            "#FFFF00",  # Yellow
            "#008000",
            "#0000FF",  # Green to blue
            "#4B0082",
            "#EE82EE",  # Indigo to violet
            "#FF00FF",  # Fuchsia
        ]
        color_transition_steps = steps // (len(colors) - 1)

        for i in range(len(colors) - 1):
            start_color = colors[i]
            end_color = colors[i + 1]
            for step in range(color_transition_steps):
                # Calculate the intermediate color
                start_rgb = tuple(int(start_color[j : j + 2], 16) for j in (1, 3, 5))
                end_rgb = tuple(int(end_color[j : j + 2], 16) for j in (1, 3, 5))
                interp_rgb = tuple(
                    start + (end - start) * step // (color_transition_steps - 1)
                    for start, end in zip(start_rgb, end_rgb)
                )
                spectrum.append("#" + "".join(f"{value:02x}" for value in interp_rgb))

        return spectrum

    def run(self):
        """
        Run the main event loop of the GUI.
        """
        self.mainloop()


# Run the main GUI window
if __name__ == "__main__":
    gui = UnicodeCharacterGeneratorGUI()
    gui.run()
