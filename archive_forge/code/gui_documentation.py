import asyncio
import io
import tkinter as tk
import threading
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from cryptography.fernet import Fernet
from core_services import LoggingManager, EncryptionManager
from database import (
from image_processing import (

        Updates the status label with a message, optionally marking it as an error.

        Args:
            message (str): The message to display.
            error (bool): If True, the message is treated as an error.
        