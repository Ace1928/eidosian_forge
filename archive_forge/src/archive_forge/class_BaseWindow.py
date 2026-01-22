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
class BaseWindow:

    def __init__(self, title, close=None):
        self.title = title
        if close:
            self.win.protocol('WM_DELETE_WINDOW', close)
        else:
            self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.things = []
        self.exists = True

    def close(self):
        self.win.destroy()
        self.exists = False

    def title(self, txt):
        self.win.title(txt)
    title = property(None, title)

    def add(self, stuff, anchor='w'):
        if isinstance(stuff, str):
            stuff = Label(stuff)
        elif isinstance(stuff, list):
            stuff = Row(stuff)
        stuff.pack(self.win, anchor=anchor)
        self.things.append(stuff)