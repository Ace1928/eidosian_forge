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
def _init_matrix(self, root):
    cframe = Frame(root, border=2, relief='sunken')
    cframe.pack(expand=0, fill='none', padx=1, pady=3, side='top')
    self._canvas = Canvas(cframe, width=200, height=200, background='white')
    self._canvas.pack(expand=0, fill='none')