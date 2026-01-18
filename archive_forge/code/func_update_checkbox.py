import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def update_checkbox(self):
    if self.checkbox:
        self.checkbox_var.set(self.get_value())