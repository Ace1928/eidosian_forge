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
def _analyze_edge(self, edge):
    """
        Given a new edge, recalculate:

            - _text_height
            - _unitsize (if the edge text is too big for the current
              _unitsize, then increase _unitsize)
        """
    c = self._chart_canvas
    if isinstance(edge, TreeEdge):
        lhs = edge.lhs()
        rhselts = []
        for elt in edge.rhs():
            if isinstance(elt, Nonterminal):
                rhselts.append(str(elt.symbol()))
            else:
                rhselts.append(repr(elt))
        rhs = ' '.join(rhselts)
    else:
        lhs = edge.lhs()
        rhs = ''
    for s in (lhs, rhs):
        tag = c.create_text(0, 0, text=s, font=self._boldfont, anchor='nw', justify='left')
        bbox = c.bbox(tag)
        c.delete(tag)
        width = bbox[2]
        edgelen = max(edge.length(), 1)
        self._unitsize = max(self._unitsize, width / edgelen)
        self._text_height = max(self._text_height, bbox[3] - bbox[1])