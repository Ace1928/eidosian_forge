from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _make_expanded_tree(self, canvas, t, key):
    make_node = self._make_node
    make_leaf = self._make_leaf
    if isinstance(t, Tree):
        node = make_node(canvas, t.label(), **self._nodeattribs)
        self._nodes.append(node)
        children = t
        subtrees = [self._make_expanded_tree(canvas, children[i], key + (i,)) for i in range(len(children))]
        treeseg = TreeSegmentWidget(canvas, node, subtrees, color=self._line_color, width=self._line_width)
        self._expanded_trees[key] = treeseg
        self._keys[treeseg] = key
        return treeseg
    else:
        leaf = make_leaf(canvas, t, **self._leafattribs)
        self._leaves.append(leaf)
        return leaf