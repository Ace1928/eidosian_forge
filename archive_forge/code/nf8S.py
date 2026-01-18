import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip

# Constants for default settings
DEFAULT_SETTINGS_FILE = "chat_settings.json"

# Path for saving screenshots and conversation log
SCREENSHOT_PATH = "screenshots"
LOG_PATH = "conversation_logs"
CONVERSATION_LOG_FILE = os.path.join(LOG_PATH, "conversation_log.txt")

# Ensure screenshot and log directories exist
os.makedirs(SCREENSHOT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


class ChatAutomation:
    def __init__(self, root):
        self.root = root
        self.mouse_position = StringVar(self.root, value="Mouse Position: (0, 0)")
        self.initialize_ui()

    def initialize_ui(self):
        """
        Initialize the user interface for setting up chat window coordinates and interaction areas.
        """
        self.root.title("Chat Automation Setup")
        self.setup_labels_entries()
        self.setup_buttons()
        self.display_mouse_position()

    def setup_labels_entries(self):
        """
        Setup labels and entry widgets for user input on coordinates and areas.
        """
        self.label_mouse_position = Label(self.root, textvariable=self.mouse_position)
        self.label_chat1 = Label(self.root, text="Chat Window 1 Coordinates (x, y):")
        self.entry_chat1_x = Entry(self.root)
        self.entry_chat1_y = Entry(self.root)
        self.label_chat2 = Label(self.root, text="Chat Window 2 Coordinates (x, y):")
        self.entry_chat2_x = Entry(self.root)
        self.entry_chat2_y = Entry(self.root)
        self.label_area = Label(self.root, text="Capture Area (width, height):")
        self.entry_area_width = Entry(self.root)
        self.entry_area_height = Entry(self.root)

        # Layout the labels and entries in the UI
        self.label_mouse_position.grid(row=0, column=0, columnspan=3)
        self.label_chat1.grid(row=1, column=0)
        self.entry_chat1_x.grid(row=1, column=1)
        self.entry_chat1_y.grid(row=1, column=2)
        self.label_chat2.grid(row=2, column=0)
        self.entry_chat2_x.grid(row=2, column=1)
        self.entry_chat2_y.grid(row=2, column=2)
        self.label_area.grid(row=3, column=0)
        self.entry_area_width.grid(row=3, column=1)
        self.entry_area_height.grid(row=3, column=2)

    def setup_buttons(self):
        """
        Setup buttons for starting the automation and saving settings.
        """
        self.start_button = Button(
            self.root, text="Start Automation", command=self.start_automation
        )
        self.save_button = Button(
            self.root, text="Save Settings", command=self.save_settings
        )
        self.start_button.grid(row=4, column=1)
        self.save_button.grid(row=4, column=2)

    def display_mouse_position(self):
        """
        Continuously update and display the mouse position in real-time.
        """
        x, y = pyautogui.position()
        self.mouse_position.set(f"Mouse Position: ({x}, {y})")
        self.root.after(100, self.display_mouse_position)  # Update position every 100ms

    def start_automation(self):
        """
        Start the automation process based on the user-defined settings.
        """
        settings = {
            "chat_window_1": {
                "coords": (int(self.entry_chat1_x.get()), int(self.entry_chat1_y.get()))
            },
            "chat_window_2": {
                "coords": (int(self.entry_chat2_x.get()), int(self.entry_chat2_y.get()))
            },
            "capture_area": (
                int(self.entry_area_width.get()),
                int(self.entry_area_height.get()),
            ),
        }
        self.automated_conversation(settings)

    def save_settings(self):
        """
        Save the user-defined settings to a JSON file.
        """
        settings = {
            "chat_window_1": {
                "coords": (int(self.entry_chat1_x.get()), int(self.entry_chat1_y.get()))
            },
            "chat_window_2": {
                "coords": (int(self.entry_chat2_x.get()), int(self.entry_chat2_y.get()))
            },
            "capture_area": (
                int(self.entry_area_width.get()),
                int(self.entry_area_height.get()),
            ),
        }
        with open(DEFAULT_SETTINGS_FILE, "w") as file:
            json.dump(settings, file)
            self.log_message(
                "Settings saved", settings, "Settings have been written to file"
            )

    def log_message(self, action, coords, message):
        """
        Log the action taken by the script with a timestamp.
        """
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(CONVERSATION_LOG_FILE, "a") as f:
            f.write(f"{timestamp} {action} at {coords}: {message}\n")

    def main_loop(self, settings):
        """
        Main loop to facilitate the automated conversation between two chat windows.
        """
        chat_window1 = settings["chat_window_1"]["coords"]
        chat_window2 = settings["chat_window_2"]["coords"]
        initial_message = "Starting automated conversation..."
        self.initiate_conversation(chat_window1, initial_message)
        # Additional logic for automated conversation


if __name__ == "__main__":
    root = Tk()
    app = ChatAutomation(root)
    root.mainloop()
