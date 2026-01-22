from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _make_collapsed_trees(self, canvas, t, key):
    if not isinstance(t, Tree):
        return
    make_node = self._make_node
    make_leaf = self._make_leaf
    node = make_node(canvas, t.label(), **self._nodeattribs)
    self._nodes.append(node)
    leaves = [make_leaf(canvas, l, **self._leafattribs) for l in t.leaves()]
    self._leaves += leaves
    treeseg = TreeSegmentWidget(canvas, node, leaves, roof=1, color=self._roof_color, fill=self._roof_fill, width=self._line_width)
    self._collapsed_trees[key] = treeseg
    self._keys[treeseg] = key
    treeseg.hide()
    for i in range(len(t)):
        child = t[i]
        self._make_collapsed_trees(canvas, child, key + (i,))