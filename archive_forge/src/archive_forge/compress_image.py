import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import zstandard as zstd
from PIL import Image
from pathlib import Path


class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compressor & Decompressor")
        self.root.geometry("400x200")

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        # File selection button
        self.file_path_var = tk.StringVar()
        file_select_btn = tk.Button(
            self.root, text="Select Image", command=self.select_file
        )
        file_select_btn.pack(pady=10)

        # Compression format dropdown
        self.format_var = tk.StringVar()
        formats = ["zstd"]  # Add more formats if available
        format_dropdown = ttk.Combobox(
            self.root, textvariable=self.format_var, values=formats
        )
        format_dropdown.pack(pady=10)
        format_dropdown.current(0)

        # Compress button
        compress_btn = tk.Button(self.root, text="Compress", command=self.compress)
        compress_btn.pack(pady=10)

        # Decompress button
        decompress_btn = tk.Button(
            self.root, text="Decompress", command=self.decompress
        )
        decompress_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.progress.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)

    def compress(self):
        try:
            self.update_progress(self.progress, 20)
            input_path = Path(self.file_path_var.get())
            if not input_path.exists():
                raise FileNotFoundError("The selected file does not exist.")

            with Image.open(input_path) as img:
                img_format = img.format.lower()

            with open(input_path, "rb") as file:
                data = file.read()

            cctx = zstd.ZstdCompressor()
            compressed_data = cctx.compress(data)

            output_path = input_path.with_suffix(f".{img_format}.zstd")
            with open(output_path, "wb") as file:
                file.write(compressed_data)

            print(f"Compressed file saved to: {output_path}")
        except Exception as e:
            messagebox.showerror("Compression Error", str(e))

    def decompress(self):
        try:
            self.update_progress(self.progress, 20)
            input_path = Path(self.file_path_var.get())
            if not input_path.exists() or not input_path.suffix.endswith("zstd"):
                raise FileNotFoundError(
                    "The selected file does not exist or is not a .zstd file."
                )

            dctx = zstd.ZstdDecompressor()
            with open(input_path, "rb") as compressed:
                decompressed_data = dctx.decompress(compressed.read())

            output_path = input_path.with_suffix("")  # Remove .zstd extension
            with open(output_path, "wb") as file:
                file.write(decompressed_data)

            print(f"Decompressed file saved to: {output_path}")
        except Exception as e:
            messagebox.showerror("Decompression Error", str(e))

    def update_progress(self, progress_bar, value=10):
        """Simulate progress for demonstration purposes."""
        current_value = progress_bar["value"]
        if current_value < 100:
            progress_bar["value"] = current_value + value
            self.root.after(100, self.update_progress, progress_bar, value)
        else:
            progress_bar["value"] = 0  # Reset progress bar for the next operation


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()
