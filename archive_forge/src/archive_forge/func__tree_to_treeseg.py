from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _tree_to_treeseg(canvas, t, make_node, make_leaf, tree_attribs, node_attribs, leaf_attribs, loc_attribs):
    if isinstance(t, Tree):
        label = make_node(canvas, t.label(), **node_attribs)
        subtrees = [_tree_to_treeseg(canvas, child, make_node, make_leaf, tree_attribs, node_attribs, leaf_attribs, loc_attribs) for child in t]
        return TreeSegmentWidget(canvas, label, subtrees, **tree_attribs)
    else:
        return make_leaf(canvas, t, **leaf_attribs)