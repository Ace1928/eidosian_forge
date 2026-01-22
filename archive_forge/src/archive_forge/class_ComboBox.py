import re
import sys
from collections import namedtuple
from functools import partial
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.messagebox import askokcancel as ask_question
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter.filedialog import LoadFileDialog, SaveFileDialog
from ase.gui.i18n import _
class ComboBox(Widget):

    def __init__(self, labels, values=None, callback=None):
        self.values = values or list(range(len(labels)))
        self.callback = callback
        self.creator = partial(ttk.Combobox, values=labels)

    def create(self, parent):
        widget = Widget.create(self, parent)
        widget.current(0)
        if self.callback:

            def callback(event):
                self.callback(self.value)
            widget.bind('<<ComboboxSelected>>', callback)
        return widget

    @property
    def value(self):
        return self.values[self.widget.current()]

    @value.setter
    def value(self, val):
        _set_entry_value(self.widget, val)