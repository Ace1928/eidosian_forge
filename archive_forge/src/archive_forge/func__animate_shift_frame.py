from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_shift_frame(self, frame, widget, dx):
    if frame > 0:
        self._animating_lock = 1
        widget.move(dx, 0)
        self._top.after(10, self._animate_shift_frame, frame - 1, widget, dx)
    else:
        del self._rtextwidgets[0]
        self._stackwidgets.append(widget)
        self._animating_lock = 0
        self._draw_stack_top(widget)
        self._highlight_productions()