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
def _draw_loclines(self):
    """
        Draw location lines.  These are vertical gridlines used to
        show where each location unit is.
        """
    BOTTOM = 50000
    c1 = self._tree_canvas
    c2 = self._sentence_canvas
    c3 = self._chart_canvas
    margin = ChartView._MARGIN
    self._loclines = []
    for i in range(0, self._chart.num_leaves() + 1):
        x = i * self._unitsize + margin
        if c1:
            t1 = c1.create_line(x, 0, x, BOTTOM)
            c1.tag_lower(t1)
        if c2:
            t2 = c2.create_line(x, 0, x, self._sentence_height)
            c2.tag_lower(t2)
        t3 = c3.create_line(x, 0, x, BOTTOM)
        c3.tag_lower(t3)
        t4 = c3.create_text(x + 2, 0, text=repr(i), anchor='nw', font=self._font)
        c3.tag_lower(t4)
        if i % 2 == 0:
            if c1:
                c1.itemconfig(t1, fill='gray60')
            if c2:
                c2.itemconfig(t2, fill='gray60')
            c3.itemconfig(t3, fill='gray60')
        else:
            if c1:
                c1.itemconfig(t1, fill='gray80')
            if c2:
                c2.itemconfig(t2, fill='gray80')
            c3.itemconfig(t3, fill='gray80')