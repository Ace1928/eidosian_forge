import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, Canvas
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
        self.initialize_ui()

    def initialize_ui(self):
        """
        Initialize the user interface for setting up chat window coordinates and interaction areas.
        """
        self.root.title("Chat Automation Setup")
        self.setup_labels_entries()
        self.setup_buttons()
        self.setup_canvas()

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

        # Layout the labels and entries in the UI
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
        Setup buttons for starting the automation and saving settings.
        """
        self.start_button = Button(
            self.root, text="Start Automation", command=self.start_automation
        )
        self.save_button = Button(
            self.root, text="Save Settings", command=self.save_settings
        )
        self.start_button.grid(row=3, column=1)
        self.save_button.grid(row=3, column=2)

    def setup_canvas(self):
        """
        Setup a canvas for user interaction to define areas for chat windows and buttons.
        """
        self.canvas = Canvas(self.root, width=500, height=500, bg="white")
        self.canvas.grid(row=4, columnspan=3)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self._drawn = None

    def on_button_press(self, event):
        """
        Handle the mouse button press event, initiating the area selection.
        """
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="black"
        )

    def on_move_press(self, event):
        """
        Handle the mouse move event with button pressed to adjust the area selection.
        """
        curX, curY = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        """
        Handle the mouse button release event, finalizing the area selection.
        """
        if self.rect:
            self.canvas.itemconfig(self.rect, outline="green")
            self.log_message(
                "Area selected",
                (self.start_x, self.start_y, event.x, event.y),
                "Area finalized",
            )

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

    def automated_conversation(self, settings):
        """
        Automate the conversation between two chat windows based on user-defined settings.
        """
        chat_window1 = settings["chat_window_1"]["coords"]
        chat_window2 = settings["chat_window_2"]["coords"]
        initial_message = "Starting automated conversation..."
        self.initiate_conversation(chat_window1, initial_message)
        # Additional logic for automated conversation

    def initiate_conversation(self, chat_coords, message):
        """
        Start the conversation in the specified chat window with the provided message.
        """
        pyautogui.click(chat_coords)  # Focus the chat window
        pyautogui.typewrite(message, interval=0.05)  # Type the message
        pyautogui.press("enter")  # Send the message
        self.log_message("Sent message", chat_coords, message)

    def log_message(self, action, coords, message):
        """
        Log the action taken by the script with a timestamp.
        """
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(CONVERSATION_LOG_FILE, "a") as f:
            f.write(f"{timestamp} {action} at {coords}: {message}\n")

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

    def message_processing_pipeline(self, image_path):
        """
        Process the captured message image and generate a response.
        """
        # Placeholder for actual image processing and response generation
        return "Automated response based on image processing and AI logic."

    def copy_to_clipboard(self):
        """
        Copy the provided text to the clipboard. This is easily facilitated by the copy button that can be clicked with the mouse in the chat interface area and will be defined at program start.

        Args:
            chat_copy_coords1: The coordinates of the copy button in the chat interface for window 1.
            chat_copy_coords2: The coordinates of the paste button in the chat interface for window 2.

        Returns:
            text: The text that is stored in the clipboard.
        """
        pyautogui.moveTo(
            chat_copy_coords
        )  # move mouse to copy button, location defined by user at program setup
        pyautogui.click()  # click the copy button
        text = pyperclip.paste()  # store copied text in variable
        self.log_message("Copied text to clipboard", chat_copy_coords, None)
        # Paste text into conversations.txt log file
        with open("conversations.txt", "a") as f:
            f.write(text + "\n")

    def paste_from_clipboard(self):
        """
        Paste the text from the clipboard. This is easily facilitated by the paste shortcut (Ctrl+V) that can be triggered to paste the text from the clipboard into the other chat window.

        Args:
            chat_paste_coords1: The coordinates of the text input area in the chat interface for window 1.
            chat_paste_coords2: The coordinates of the text input area in the chat interface for window 2.

        Returns:
            None
        """
        pyautogui.moveTo(chat_paste_coords)
        pyautogui.click()
        pyautogui.hotkey("ctrl", "v")
        self.log_message("Pasted text from clipboard", chat_paste_coords, None)

    def stop_or_not(self):
        """
        Determine the operational status based on the visibility of the stop button within the chat interface, utilizing advanced image processing techniques to ensure precise and accurate detection, thereby dictating the subsequent actions of the automation script.

        Args:
            chat_stop_coords1: The coordinates of the stop button in the chat interface for window 1.
            chat_stop_coords2: The coordinates of the stop button in the chat interface for window 2.

        Returns:
            None
        """
        # Capture a screenshot of the designated area where the stop button is expected to be located
        screenshot = pyautogui.screenshot(
            region=(chat_stop_coords[0], chat_stop_coords[1], *capture_area)
        )
        screenshot.save(os.path.join(SCREENSHOT_PATH, "screenshot.png"))

        # Load the screenshot for analysis using OpenCV
        screenshot_image = cv2.imread(os.path.join(SCREENSHOT_PATH, "screenshot.png"))

        # Engage the user in defining the precise area for stop button detection
        print(
            "Please delineate the area where the stop button is anticipated to manifest."
        )
        user_selected_area = (
            self.select_area_on_screen()
        )  # This method captures a user-defined screen area

        # Dispatch a meticulously crafted message to the chat to induce the appearance of the stop button
        self.send_message_to_chat(
            "Initiate count sequence from 1 to 100 with a 0.5-second interval using Python, displaying each count in real-time."
        )

        # Allow a brief pause to ensure the chat processes the message and the stop button appears
        time.sleep(3)

        # Capture a new screenshot specifically of the area where the stop button is expected to appear post-message
        stop_button_screenshot = pyautogui.screenshot(region=user_selected_area)
        stop_button_screenshot.save(
            os.path.join(SCREENSHOT_PATH, "stop_button_template.png")
        )

        # Load the stop button image as a template for matching
        stop_button_template = cv2.imread(
            os.path.join(SCREENSHOT_PATH, "stop_button_template.png")
        )
        template_width, template_height = stop_button_template.shape[:-1]

        # Execute template matching to locate the stop button within the loaded screenshot
        match_result = cv2.matchTemplate(
            screenshot_image, stop_button_template, cv2.TM_CCOEFF_NORMED
        )
        detection_threshold = 0.8  # Set a high confidence threshold for detection
        locations = np.where(match_result >= detection_threshold)

        # Evaluate the presence of the stop button based on the template matching results
        stop_button_visible = False
        for point in zip(*locations[::-1]):  # Iterate through found locations
            cv2.rectangle(
                screenshot_image,
                point,
                (point[0] + template_width, point[1] + template_height),
                (0, 255, 0),
                2,
            )
            stop_button_visible = True

        # Log the detection process and outcome
        self.log_message(
            "Detection of Stop button concluded",
            chat_stop_coords,
            "Stop button visible" if stop_button_visible else "Stop button not visible",
        )

        # Optionally, save the annotated image with detection results for verification
        cv2.imwrite(
            os.path.join(SCREENSHOT_PATH, "screenshot_with_detection.png"),
            screenshot_image,
        )

        if stop_button_visible:
            # If the stop button is detected and resembles the expected design (e.g., square within a circle), continue monitoring the chat output.
            self.log_message(
                "Stop button detected, monitoring ongoing", chat_stop_coords, None
            )
        else:
            # If no valid stop button is detected, proceed to copy the chat message to the clipboard and paste it into the alternate chat window.
            self.copy_to_clipboard()
            self.paste_from_clipboard()
            self.log_message(
                "No stop button detected, message copied and pasted to alternate chat",
                chat_stop_coords,
                None,
            )

        # Continuously monitor the chat response based on the stop button's status, facilitating an ongoing conversation by alternating between copying and pasting chat outputs.

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
    settings = app.load_settings()
    app.main_loop(settings)
