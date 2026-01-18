import sys
import os
import tkinter as tk
from tkinter import ttk
def zoom_out(self):
    l = self.slider.left_end
    r = self.slider.right_end
    if r - l > self.max_span:
        return
    self.slider.left_end = 0.5 * (3.0 * l - r)
    self.slider.right_end = 0.5 * (3.0 * r - l)
    self.slider.set_value(self.current_value)
    self._update_labels()