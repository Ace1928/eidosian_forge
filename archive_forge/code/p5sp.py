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

        if os.path.exists(DEFAULT_SETTINGS_FILE):
            with open(DEFAULT_SETTINGS_FILE, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(
                "Settings file not found. Please configure the settings."
            )

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

    def screenshot_stop_button(self):
        """
        Capture a screenshot of the stop button in the chat interface. This is used to determine if the conversation has ended and the script should stop.

        Args:
            chat_stop_coords1: The coordinates of the stop button in the chat interface for window 1.
            chat_stop_coords2: The coordinates of the stop button in the chat interface for window 2.

        Returns:
            None
        """
        screenshot = pyautogui.screenshot(
            region=(chat_stop_coords[0], chat_stop_coords[1], *capture_area)
        )
        screenshot.save('screenshot.png')
        # Analyse the screenshot to determine if the stop button is visible
        # Utilizing advanced image processing techniques to analyze the screenshot for the presence of a stop button
        # Load the screenshot image from memory
        screenshot_image = cv2.imread('screenshot.png')
        
        # Prompt the user to select the area where the stop button will be located
        print("Please select the area where the stop button will appear.")
        user_selected_area = self.select_area_on_screen()  # Assuming select_area_on_screen is a method to capture user-defined screen area

        # Send a specific message to the chat to trigger the stop button appearance
        self.send_message_to_chat("Count to 100 in your code interpreter using python with a 0.5 second delay between each number, executing authentically and outputting directly to console in real time for me to view")
        
        # Wait for 3 seconds before taking a screenshot to ensure the message is processed
        time.sleep(3)
        
        # Capture the screenshot of the selected area where the stop button is expected to appear
        stop_button_screenshot = pyautogui.screenshot(region=user_selected_area)
        stop_button_screenshot.save('stop_button_template.png')
        
        # Load the captured image as the template for the stop button
        stop_button_template = cv2.imread('stop_button_template.png')
        w, h = stop_button_template.shape[:-1]

        # Apply template matching to find the stop button in the screenshot
        res = cv2.matchTemplate(screenshot_image, stop_button_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Define a threshold for detection confidence
        loc = np.where(res >= threshold)
        
        # Determine if the stop button is visible based on the presence of matching points
        stop_button_visible = False
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            cv2.rectangle(screenshot_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            stop_button_visible = True
        
        # Log the result of the detection
        self.log_message("Stop button detection completed", chat_stop_coords, stop_button_visible)
        
        # Save the result image for verification (optional)
        cv2.imwrite('screenshot_with_detection.png', screenshot_image)
        
        if stop_button_visible:
            # if button looks like a stop button (square inside a circle) then continue waiting for the output to finish in that chat window.
            
        else:
            # if button not a stop button then copy the message to the clipboard and paste it into the other chat window.
            

        # wait for the response of the chat window to finish (based on the stop button) and then continue the conversation by copying the output, going to the other window, pasting, waiting for it to finish, repeating ad infinitum.

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
