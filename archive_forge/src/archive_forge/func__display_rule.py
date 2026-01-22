import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
def _display_rule(self, rule):
    if rule is None:
        self._rulelabel2['text'] = ''
    else:
        name = str(rule)
        self._rulelabel2['text'] = name
        size = self._cv.get_font_size()