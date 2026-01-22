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
class ASEFileChooser(LoadFileDialog):

    def __init__(self, win, formatcallback=lambda event: None):
        from ase.io.formats import all_formats, get_ioformat
        LoadFileDialog.__init__(self, win, _('Open ...'))
        labels = [_('Automatic')]
        values = ['']

        def key(item):
            return item[1][0]
        for format, (description, code) in sorted(all_formats.items(), key=key):
            io = get_ioformat(format)
            if io.can_read and description != '?':
                labels.append(_(description))
                values.append(format)
        self.format = None

        def callback(value):
            self.format = value
        Label(_('Choose parser:')).pack(self.top)
        formats = ComboBox(labels, values, callback)
        formats.pack(self.top)