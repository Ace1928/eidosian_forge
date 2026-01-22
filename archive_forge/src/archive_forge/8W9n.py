import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry

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
        self.initialize_ui()

    def initialize_ui(self):
        """
        Initialize the user interface for setting up chat window coordinates and interaction areas.
        """
        self.root.title("Chat Automation Setup")
        self.setup_labels_entries()
        self.setup_buttons()

    def setup_labels_entries(self):
        """
        Setup labels and entry widgets for user input on coordinates and areas.
        """
        self.label_chat1 = Label(self.root, text="Chat Window 1 Coordinates (x, y):")
        self.entry_chat1_x = Entry(self.root)
        self.entry_chat1_y = Entry(self.root)
        self.label_chat2 = Label(self.root, text="Chat Window 2 Coordinates (x, y):")
        self.entry_chat2_x = Entry(self.root)
        self.entry_chat2_y = Entry(self.root)
        self.label_area = Label(self.root, text="Capture Area (width, height):")
        self.entry_area_width = Entry(self.root)
        self.entry_area_height = Entry(self.root)

        self.label_chat1.grid(row=0, column=0)
        self.entry_chat1_x.grid(row=0, column=1)
        self.entry_chat1_y.grid(row=0, column=2)
        self.label_chat2.grid(row=1, column=0)
        self.entry_chat2_x.grid(row=1, column=1)
        self.entry_chat2_y.grid(row=1, column=2)
        self.label_area.grid(row=2, column=0)
        self.entry_area_width.grid(row=2, column=1)
        self.entry_area_height.grid(row=2, column=2)

    def setup_buttons(self):
        """
        Setup buttons for saving settings and starting the automation.
        """
        self.button_save = Button(
            self.root, text="Save Settings", command=self.save_settings
        )
        self.button_start = Button(
            self.root, text="Start Automation", command=self.start_automation
        )
        self.button_save.grid(row=3, column=1)
        self.button_start.grid(row=3, column=2)

    def save_settings(self):
        """
        Save the user-defined settings for chat coordinates and areas.
        """
        settings = {
            "chat_window_1": {
                "coords": (
                    int(self.entry_chat1_x.get()),
                    int(self.entry_chat1_y.get()),
                ),
                "capture_area": (
                    int(self.entry_area_width.get()),
                    int(self.entry_area_height.get()),
                ),
            },
            "chat_window_2": {
                "coords": (
                    int(self.entry_chat2_x.get()),
                    int(self.entry_chat2_y.get()),
                ),
                "capture_area": (
                    int(self.entry_area_width.get()),
                    int(self.entry_area_height.get()),
                ),
            },
        }
        with open(DEFAULT_SETTINGS_FILE, "w") as file:
            json.dump(settings, file)
        print("Settings saved successfully.")

    def start_automation(self):
        """
        Start the chat automation using the saved settings.
        """
        settings = self.load_settings()
        self.main_loop(settings)

    def load_settings(self):
        """
        Load chat window settings from a JSON file.
        """
        if os.path.exists(DEFAULT_SETTINGS_FILE):
            with open(DEFAULT_SETTINGS_FILE, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(
                "Settings file not found. Please configure the settings."
            )

    def main_loop(self, settings):
        """
        Main loop to facilitate the automated conversation between two chat windows.
        """
        chat_window1 = settings["chat_window_1"]["coords"]
        chat_window2 = settings["chat_window_2"]["coords"]
        initial_message = "Starting automated conversation..."
        self.initiate_conversation(chat_window1, initial_message)
        # Additional logic for automated conversation

    def load_settings(self):
        """
        Load chat window settings from a JSON file.
        """
        if os.path.exists(DEFAULT_SETTINGS_FILE):
            with open(DEFAULT_SETTINGS_FILE, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(
                "Settings file not found. Please configure the settings."
            )

    def setup_ui(self):
        """
        Setup the user interface for defining coordinates and areas.
        """
        # [Insert detailed Tkinter UI setup code here, including labels, entry boxes for coordinates, and save/setup buttons]

    def initiate_conversation(self, chat_coords, message):
        """
        Start the conversation in the specified chat window with the provided message.
        """
        pyautogui.click(chat_coords)  # Focus the chat window
        pyautogui.typewrite(message, interval=0.05)  # Type the message
        pyautogui.press("enter")  # Send the message
        self.log_message("Sent message", chat_coords, message)

    def capture_screenshot(self, chat_coords, capture_area, prefix="chat"):
        """
        Capture a screenshot of the chat window and save it with a unique timestamped filename.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_PATH, filename)
        screenshot = pyautogui.screenshot(
            region=(chat_coords[0], chat_coords[1], *capture_area)
        )
        screenshot.save(filepath)
        self.log_message("Screenshot captured", chat_coords, filepath)
        return filepath

    def log_message(self, action, coords, message):
        """
        Log the action taken by the script with a timestamp.
        """
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(CONVERSATION_LOG_FILE, "a") as f:
            f.write(f"{timestamp} {action} at {coords}: {message}\n")

    def message_processing_pipeline(self, image_path):
        """
        Process the captured message image and generate a response.
        """
        # Placeholder for actual image processing and response generation
        return "Automated response based on image processing and AI logic."

    def main_loop(self, settings):
        """
        Main loop to facilitate the automated conversation between two chat windows.
        """
        chat_window1 = settings["chat_window_1"]
        chat_window2 = settings["chat_window_2"]
        initial_message = settings["initial_message"]

        self.initiate_conversation(chat_window1["coords"], initial_message)

        while True:
            response_image_path = self.capture_screenshot(
                chat_window2["coords"], chat_window2["capture_area"], "chat2"
            )
            response_message = self.message_processing_pipeline(response_image_path)

            if "end" in response_message.lower():
                self.log_message(
                    "Conversation end detected",
                    chat_window2["coords"],
                    response_message,
                )
                break

            self.initiate_conversation(chat_window1["coords"], response_message)
            time.sleep(random.uniform(1.5, 3.0))

            response_image_path = self.capture_screenshot(
                chat_window1["coords"], chat_window1["capture_area"], "chat1"
            )
            response_message = self.message_processing_pipeline(response_image_path)
            self.initiate_conversation(chat_window2["coords"], response_message)
            time.sleep(random.uniform(1.5, 3.0))


if __name__ == "__main__":
    root = Tk()
    app = ChatAutomation(root)
    root.mainloop()
    settings = app.load_settings()
    app.main_loop(settings)
