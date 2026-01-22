from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _highlight_hover(self, event):
    index = self._prodlist.nearest(event.y)
    if self._hover == index:
        return
    self._clear_hover()
    selection = [int(s) for s in self._prodlist.curselection()]
    if index in selection:
        rhslen = len(self._productions[index].rhs())
        for stackwidget in self._stackwidgets[-rhslen:]:
            if isinstance(stackwidget, TreeSegmentWidget):
                stackwidget.label()['color'] = '#00a000'
            else:
                stackwidget['color'] = '#00a000'
    self._hover = index