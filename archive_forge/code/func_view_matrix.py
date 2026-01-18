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
def view_matrix(self, *e):
    if self._matrix is not None:
        self._matrix.destroy()
    self._matrix = ChartMatrixView(self._root, self._chart)
    self._matrix.add_callback('select', self._select_matrix_edge)