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
def _draw_sentence(self):
    """Draw the sentence string."""
    if self._chart.num_leaves() == 0:
        return
    c = self._sentence_canvas
    margin = ChartView._MARGIN
    y = ChartView._MARGIN
    for i, leaf in enumerate(self._chart.leaves()):
        x1 = i * self._unitsize + margin
        x2 = x1 + self._unitsize
        x = (x1 + x2) / 2
        tag = c.create_text(x, y, text=repr(leaf), font=self._font, anchor='n', justify='left')
        bbox = c.bbox(tag)
        rt = c.create_rectangle(x1 + 2, bbox[1] - ChartView._LEAF_SPACING / 2, x2 - 2, bbox[3] + ChartView._LEAF_SPACING / 2, fill='#f0f0f0', outline='#f0f0f0')
        c.tag_lower(rt)