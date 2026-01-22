from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_expand_frame(self, widget, colors):
    if len(colors) > 0:
        self._animating_lock = 1
        widget['color'] = colors[0]
        for subtree in widget.subtrees():
            if isinstance(subtree, TreeSegmentWidget):
                subtree.label()['color'] = colors[0]
            else:
                subtree['color'] = colors[0]
        self._top.after(50, self._animate_expand_frame, widget, colors[1:])
    else:
        widget['color'] = 'black'
        for subtree in widget.subtrees():
            if isinstance(subtree, TreeSegmentWidget):
                subtree.label()['color'] = 'black'
            else:
                subtree['color'] = 'black'
        self._redraw_quick()
        widget.label()['color'] = 'black'
        self._animating_lock = 0
        if self._autostep:
            self._step()