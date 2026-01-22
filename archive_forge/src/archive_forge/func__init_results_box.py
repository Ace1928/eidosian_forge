import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_results_box(self, parent):
    innerframe = Frame(parent)
    i1 = Frame(innerframe)
    i2 = Frame(innerframe)
    vscrollbar = Scrollbar(i1, borderwidth=1)
    hscrollbar = Scrollbar(i2, borderwidth=1, orient='horiz')
    self.results_box = Text(i1, font=Font(family='courier', size='16'), state='disabled', borderwidth=1, yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set, wrap='none', width='40', height='20', exportselection=1)
    self.results_box.pack(side='left', fill='both', expand=True)
    self.results_box.tag_config(self._HIGHLIGHT_WORD_TAG, foreground=self._HIGHLIGHT_WORD_COLOUR)
    self.results_box.tag_config(self._HIGHLIGHT_LABEL_TAG, foreground=self._HIGHLIGHT_LABEL_COLOUR)
    vscrollbar.pack(side='left', fill='y', anchor='e')
    vscrollbar.config(command=self.results_box.yview)
    hscrollbar.pack(side='left', fill='x', expand=True, anchor='w')
    hscrollbar.config(command=self.results_box.xview)
    Label(i2, text='   ', background=self._BACKGROUND_COLOUR).pack(side='left', anchor='e')
    i1.pack(side='top', fill='both', expand=True, anchor='n')
    i2.pack(side='bottom', fill='x', anchor='s')
    innerframe.pack(side='top', fill='both', expand=True)