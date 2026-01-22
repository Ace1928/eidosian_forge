import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _selectprod_cb(self, production):
    canvas = self._treelet_canvas
    self._prodlist.highlight(production)
    if self._treelet is not None:
        self._treelet.destroy()
    rhs = production.rhs()
    for i, elt in enumerate(rhs):
        if isinstance(elt, Nonterminal):
            elt = Tree(elt)
    tree = Tree(production.lhs().symbol(), *rhs)
    fontsize = int(self._size.get())
    node_font = ('helvetica', -(fontsize + 4), 'bold')
    leaf_font = ('helvetica', -(fontsize + 2))
    self._treelet = tree_to_treesegment(canvas, tree, node_font=node_font, leaf_font=leaf_font)
    self._treelet['draggable'] = 1
    x1, y1, x2, y2 = self._treelet.bbox()
    w, h = (int(canvas['width']), int(canvas['height']))
    self._treelet.move((w - x1 - x2) / 2, (h - y1 - y2) / 2)
    self._markproduction(production)