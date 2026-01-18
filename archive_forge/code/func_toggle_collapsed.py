from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def toggle_collapsed(self, treeseg):
    """
        Collapse/expand a tree.
        """
    old_treeseg = treeseg
    if old_treeseg['roof']:
        new_treeseg = self._expanded_trees[self._keys[old_treeseg]]
    else:
        new_treeseg = self._collapsed_trees[self._keys[old_treeseg]]
    if old_treeseg.parent() is self:
        self._remove_child_widget(old_treeseg)
        self._add_child_widget(new_treeseg)
        self._treeseg = new_treeseg
    else:
        old_treeseg.parent().replace_child(old_treeseg, new_treeseg)
    new_treeseg.show()
    newx, newy = new_treeseg.label().bbox()[:2]
    oldx, oldy = old_treeseg.label().bbox()[:2]
    new_treeseg.move(oldx - newx, oldy - newy)
    old_treeseg.hide()
    new_treeseg.parent().update(new_treeseg)