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
def _draw_treetok(self, treetok, index, depth=0):
    """
        :param index: The index of the first leaf in the tree.
        :return: The index of the first leaf after the tree.
        """
    c = self._tree_canvas
    margin = ChartView._MARGIN
    child_xs = []
    for child in treetok:
        if isinstance(child, Tree):
            child_x, index = self._draw_treetok(child, index, depth + 1)
            child_xs.append(child_x)
        else:
            child_xs.append((2 * index + 1) * self._unitsize / 2 + margin)
            index += 1
    if child_xs:
        nodex = sum(child_xs) / len(child_xs)
    else:
        nodex = (2 * index + 1) * self._unitsize / 2 + margin
        index += 1
    nodey = depth * (ChartView._TREE_LEVEL_SIZE + self._text_height)
    tag = c.create_text(nodex, nodey, anchor='n', justify='center', text=str(treetok.label()), fill='#042', font=self._boldfont)
    self._tree_tags.append(tag)
    childy = nodey + ChartView._TREE_LEVEL_SIZE + self._text_height
    for childx, child in zip(child_xs, treetok):
        if isinstance(child, Tree) and child:
            tag = c.create_line(nodex, nodey + self._text_height, childx, childy, width=2, fill='#084')
            self._tree_tags.append(tag)
        if isinstance(child, Tree) and (not child):
            tag = c.create_line(nodex, nodey + self._text_height, childx, childy, width=2, fill='#048', dash='2 3')
            self._tree_tags.append(tag)
        if not isinstance(child, Tree):
            tag = c.create_line(nodex, nodey + self._text_height, childx, 10000, width=2, fill='#084')
            self._tree_tags.append(tag)
    return (nodex, index)