from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_fonts(self, root):
    self._sysfont = Font(font=Button()['font'])
    root.option_add('*Font', self._sysfont)
    self._size = IntVar(root)
    self._size.set(self._sysfont.cget('size'))
    self._boldfont = Font(family='helvetica', weight='bold', size=self._size.get())
    self._font = Font(family='helvetica', size=self._size.get())