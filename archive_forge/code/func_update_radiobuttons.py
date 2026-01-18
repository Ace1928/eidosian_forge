import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def update_radiobuttons(self):
    if self.radio_buttons:
        self.radio_var.set(self.get_value())