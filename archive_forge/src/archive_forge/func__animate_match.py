from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_match(self, treeloc):
    widget = self._get(self._tree, treeloc)
    dy = (self._textwidgets[0].bbox()[1] - widget.bbox()[3] - 10.0) / max(1, self._animation_frames.get())
    self._animate_match_frame(self._animation_frames.get(), widget, dy)