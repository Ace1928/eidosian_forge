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
def _color_edge(self, edge, linecolor=None, textcolor=None):
    """
        Color in an edge with the given colors.
        If no colors are specified, use intelligent defaults
        (dependent on selection, etc.)
        """
    if edge not in self._edgetags:
        return
    c = self._chart_canvas
    if linecolor is not None and textcolor is not None:
        if edge in self._marks:
            linecolor = self._marks[edge]
        tags = self._edgetags[edge]
        c.itemconfig(tags[0], fill=linecolor)
        c.itemconfig(tags[1], fill=textcolor)
        c.itemconfig(tags[2], fill=textcolor, outline=textcolor)
        c.itemconfig(tags[3], fill=textcolor)
        c.itemconfig(tags[4], fill=textcolor)
        return
    else:
        N = self._chart.num_leaves()
        if edge in self._marks:
            self._color_edge(self._marks[edge])
        if edge.is_complete() and edge.span() == (0, N):
            self._color_edge(edge, '#084', '#042')
        elif isinstance(edge, LeafEdge):
            self._color_edge(edge, '#48c', '#246')
        else:
            self._color_edge(edge, '#00f', '#008')