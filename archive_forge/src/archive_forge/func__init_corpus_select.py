import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_corpus_select(self, parent):
    innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
    self.var = StringVar(innerframe)
    self.var.set(self.model.DEFAULT_CORPUS)
    Label(innerframe, justify=LEFT, text=' Corpus: ', background=self._BACKGROUND_COLOUR, padx=2, pady=1, border=0).pack(side='left')
    other_corpora = list(self.model.CORPORA.keys()).remove(self.model.DEFAULT_CORPUS)
    om = OptionMenu(innerframe, self.var, self.model.DEFAULT_CORPUS, *self.model.non_default_corpora(), command=self.corpus_selected)
    om['borderwidth'] = 0
    om['highlightthickness'] = 1
    om.pack(side='left')
    innerframe.pack(side='top', fill='x', anchor='n')