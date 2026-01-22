from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _tree_leaves(self, tree=None):
    if tree is None:
        tree = self._tree
    if isinstance(tree, TreeSegmentWidget):
        leaves = []
        for child in tree.subtrees():
            leaves += self._tree_leaves(child)
        return leaves
    else:
        return [tree]