import sys
import os
import tkinter as tk
from tkinter import ttk
def zoom_in(self):
    l = self.slider.left_end
    r = self.slider.right_end
    if r - l < self.min_span:
        return
    self.slider.left_end = 0.25 * (3.0 * l + r)
    self.slider.right_end = 0.25 * (3.0 * r + l)
    self.set_value(self.current_value)
    self._update_labels()