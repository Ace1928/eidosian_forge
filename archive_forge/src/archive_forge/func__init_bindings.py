from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_bindings(self):
    self._top.bind('<Control-q>', self.destroy)
    self._top.bind('<Control-x>', self.destroy)
    self._top.bind('<Alt-q>', self.destroy)
    self._top.bind('<Alt-x>', self.destroy)
    self._top.bind('<space>', self.step)
    self._top.bind('<s>', self.shift)
    self._top.bind('<Alt-s>', self.shift)
    self._top.bind('<Control-s>', self.shift)
    self._top.bind('<r>', self.reduce)
    self._top.bind('<Alt-r>', self.reduce)
    self._top.bind('<Control-r>', self.reduce)
    self._top.bind('<Delete>', self.reset)
    self._top.bind('<u>', self.undo)
    self._top.bind('<Alt-u>', self.undo)
    self._top.bind('<Control-u>', self.undo)
    self._top.bind('<Control-z>', self.undo)
    self._top.bind('<BackSpace>', self.undo)
    self._top.bind('<Control-p>', self.postscript)
    self._top.bind('<Control-h>', self.help)
    self._top.bind('<F1>', self.help)
    self._top.bind('<Control-g>', self.edit_grammar)
    self._top.bind('<Control-t>', self.edit_sentence)
    self._top.bind('-', lambda e, a=self._animate: a.set(20))
    self._top.bind('=', lambda e, a=self._animate: a.set(10))
    self._top.bind('+', lambda e, a=self._animate: a.set(4))