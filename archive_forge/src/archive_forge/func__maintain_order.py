from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _maintain_order(self, child):
    if self._horizontal:
        return self._maintain_order_horizontal(child)
    else:
        return self._maintain_order_vertical(child)