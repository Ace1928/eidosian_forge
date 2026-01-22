from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _maintain_order_vertical(self, child):
    left, top, right, bot = child.bbox()
    if child is self._label:
        for subtree in self._subtrees:
            x1, y1, x2, y2 = subtree.bbox()
            if bot + self._yspace > y1:
                subtree.move(0, bot + self._yspace - y1)
        return self._subtrees
    else:
        moved = [child]
        index = self._subtrees.index(child)
        x = right + self._xspace
        for i in range(index + 1, len(self._subtrees)):
            x1, y1, x2, y2 = self._subtrees[i].bbox()
            if x > x1:
                self._subtrees[i].move(x - x1, 0)
                x += x2 - x1 + self._xspace
                moved.append(self._subtrees[i])
        x = left - self._xspace
        for i in range(index - 1, -1, -1):
            x1, y1, x2, y2 = self._subtrees[i].bbox()
            if x < x2:
                self._subtrees[i].move(x - x2, 0)
                x -= x2 - x1 + self._xspace
                moved.append(self._subtrees[i])
        x1, y1, x2, y2 = self._label.bbox()
        if y2 > top - self._yspace:
            self._label.move(0, top - self._yspace - y2)
            moved = self._subtrees
    return moved