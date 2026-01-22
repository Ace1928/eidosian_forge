import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import io
import asyncio
from threading import Thread
from image_processing import (
    compress_and_encrypt_image_data,
    decrypt_and_decompress_image_data,
    get_image_hash,
    is_valid_image,
)
from database import (
    init_db,
    insert_compressed_image,
    retrieve_compressed_image,
    get_images_metadata,
    get_valid_encryption_key,
)
from cryptography.fernet import Fernet
import logging
from logging_config import configure_logging

configure_logging()


class ImageCompressorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Compressor")
        self.geometry("800x600")  # Adjust the window size as needed
        self.original_image_data = None
        self.img_hash = None
        self.cipher_suite = Fernet(get_valid_encryption_key())
        self.current_page = 0
        self.items_per_page = 10
        self.init_widgets()
        Thread(target=self.async_init_db, daemon=True).start()

    def async_init_db(self):
        asyncio.run(init_db())

    def init_widgets(self):
        self.load_btn = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_btn.pack()

        self.compress_btn = tk.Button(
            self,
            text="Compress & Store Image",
            state="disabled",
            command=self.compress_image,
        )
        self.compress_btn.pack()

        self.decompress_btn = tk.Button(
            self, text="Decompress & Show Image", command=self.decompress_and_show
        )
        self.decompress_btn.pack()

        self.browse_db_btn = tk.Button(
            self, text="Browse Stored Images", command=self.browse_stored_images
        )
        self.browse_db_btn.pack()

        self.original_img_label = tk.Label(self)
        self.original_img_label.pack(side="left", padx=10)

        self.decompressed_img_label = tk.Label(self)
        self.decompressed_img_label.pack(side="right", padx=10)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

        self.progress_bar = ttk.Progressbar(
            self, orient="horizontal", mode="indeterminate"
        )
        self.progress_bar.pack()

        # Pagination controls
        self.pagination_frame = tk.Frame(self)
        self.prev_button = tk.Button(
            self.pagination_frame, text="Previous", command=self.previous_page
        )
        self.next_button = tk.Button(
            self.pagination_frame, text="Next", command=self.next_page
        )
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.next_button.pack(side=tk.RIGHT, padx=10)
        self.pagination_frame.pack(side=tk.BOTTOM, pady=10)

        # Image display area
        self.image_display_frame = tk.Frame(self)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)
        self.image_labels = [
            tk.Label(self.image_display_frame) for _ in range(self.items_per_page)
        ]
        for label in self.image_labels:
            label.pack()

    def update_image_display(self, metadata_list):
        # Clear existing images
        for label in self.image_labels:
            label.config(image="")
            label.image = None

        # Display new images
        for metadata, label in zip(metadata_list, self.image_labels):
            try:
                img_data = metadata[0]  # Assuming metadata is a tuple (image_data, ...)
                img = Image.open(io.BytesIO(img_data))
                img.thumbnail((100, 100), Image.ANTIALIAS)  # Example thumbnail size
                tk_img = ImageTk.PhotoImage(img)
                label.config(image=tk_img)
                label.image = tk_img  # Keep a reference
            except Exception as e:
                logging.error(f"Error displaying image: {e}")

    async def async_browse_stored_images_coroutine(self):
        offset = self.current_page * self.items_per_page
        limit = self.items_per_page
        try:
            metadata_list = await get_images_metadata(offset, limit)
            self.after(0, self.update_image_display, metadata_list)
        except Exception as e:
            logging.error(f"Failed to browse stored images: {e}")
            messagebox.showerror("Error", f"Failed to browse stored images: {e}")

    def async_browse_stored_images(self):
        asyncio.run_coroutine_threadsafe(
            self.async_browse_stored_images_coroutine(), asyncio.get_event_loop()
        )

    def next_page(self):
        self.current_page += 1
        self.async_browse_stored_images()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.async_browse_stored_images()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path and self.is_valid_image(file_path):
            try:
                img = Image.open(file_path)
                img = img.convert("RGBA")
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                self.original_image_data = img_bytes.getvalue()

                tk_img = ImageTk.PhotoImage(img)
                self.original_img_label.config(image=tk_img)
                self.original_img_label.image = tk_img

                self.img_hash = get_image_hash(self.original_image_data)
                self.compress_btn.config(state="normal")
                self.status_label.config(text="Image loaded successfully.")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                messagebox.showerror(
                    "Error", "An unexpected error occurred. Please try again."
                )
        else:
            messagebox.showerror("Error", "Invalid file selected.")

    def is_valid_image(self, file_path):
        try:
            Image.open(file_path)
            return True
        except IOError:
            return False

    def compress_image(self):
        self.compress_btn.config(state="disabled")
        self.status_label.config(text="Compressing image...")
        self.progress_bar.start()
        Thread(target=self.async_compress_image, daemon=True).start()

    def async_compress_image(self):
        try:
            image_format = Image.open(io.BytesIO(self.original_image_data)).format
            encrypted_data = compress_and_encrypt_image_data(
                self.original_image_data, image_format, self.cipher_suite
            )
            asyncio.run(
                insert_compressed_image(self.img_hash, image_format, encrypted_data)
            )
            self.status_label.config(text="Image compressed and stored successfully.")
        except Exception as e:
            logging.error(f"Failed to compress and store image: {e}")
            self.status_label.config(
                text="Failed to compress and store image. Please try again."
            )
        finally:
            self.progress_bar.stop()
            self.compress_btn.config(state="normal")

    def decompress_and_show(self):
        self.progress_bar.start()
        asyncio.run_coroutine_threadsafe(
            self.async_decompress_and_show(), asyncio.get_event_loop()
        )

    async def async_decompress_and_show(self):
        try:
            encrypted_data, image_format = await retrieve_compressed_image(
                self.img_hash
            )
            decompressed_data = decrypt_and_decompress_image_data(
                encrypted_data, self.cipher_suite
            )
            img = Image.open(io.BytesIO(decompressed_data))
            tk_img = ImageTk.PhotoImage(img)
            self.decompressed_img_label.config(image=tk_img)
            self.decompressed_img_label.image = tk_img
            self.status_label.config(text="Image decompressed successfully.")
        except Exception as e:
            logging.error(f"Failed to decompress image: {e}")
            self.status_label.config(
                text="Failed to decompress image. Please try again."
            )
        finally:
            self.progress_bar.stop()

    def browse_stored_images(self):
        self.current_page = 0  # Reset to the first page
        self.async_browse_stored_images()


if __name__ == "__main__":
    app = ImageCompressorApp()
    app.mainloop()
