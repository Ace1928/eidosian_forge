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
def _draw_edge(self, edge, lvl):
    """
        Draw a single edge on the ChartView.
        """
    c = self._chart_canvas
    x1 = edge.start() * self._unitsize + ChartView._MARGIN
    x2 = edge.end() * self._unitsize + ChartView._MARGIN
    if x2 == x1:
        x2 += max(4, self._unitsize / 5)
    y = (lvl + 1) * self._chart_level_size
    linetag = c.create_line(x1, y, x2, y, arrow='last', width=3)
    if isinstance(edge, TreeEdge):
        rhs = []
        for elt in edge.rhs():
            if isinstance(elt, Nonterminal):
                rhs.append(str(elt.symbol()))
            else:
                rhs.append(repr(elt))
        pos = edge.dot()
    else:
        rhs = []
        pos = 0
    rhs1 = ' '.join(rhs[:pos])
    rhs2 = ' '.join(rhs[pos:])
    rhstag1 = c.create_text(x1 + 3, y, text=rhs1, font=self._font, anchor='nw')
    dotx = c.bbox(rhstag1)[2] + 6
    doty = (c.bbox(rhstag1)[1] + c.bbox(rhstag1)[3]) / 2
    dottag = c.create_oval(dotx - 2, doty - 2, dotx + 2, doty + 2)
    rhstag2 = c.create_text(dotx + 6, y, text=rhs2, font=self._font, anchor='nw')
    lhstag = c.create_text((x1 + x2) / 2, y, text=str(edge.lhs()), anchor='s', font=self._boldfont)
    self._edgetags[edge] = (linetag, rhstag1, dottag, rhstag2, lhstag)

    def cb(event, self=self, edge=edge):
        self._fire_callbacks('select', edge)
    c.tag_bind(rhstag1, '<Button-1>', cb)
    c.tag_bind(rhstag2, '<Button-1>', cb)
    c.tag_bind(linetag, '<Button-1>', cb)
    c.tag_bind(dottag, '<Button-1>', cb)
    c.tag_bind(lhstag, '<Button-1>', cb)
    self._color_edge(edge)