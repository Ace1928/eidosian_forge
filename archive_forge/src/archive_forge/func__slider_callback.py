import sys
import os
import tkinter as tk
from tkinter import ttk
def _slider_callback(self, value):
    self.current_value = value
    self._update_labels()
    if self.callback:
        self.callback(value)
    if self.on_change:
        self.on_change()