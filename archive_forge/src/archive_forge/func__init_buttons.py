from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_buttons(self, parent):
    self._buttonframe = buttonframe = Frame(parent)
    buttonframe.pack(fill='none', side='bottom')
    Button(buttonframe, text='Step', background='#90c0d0', foreground='black', command=self.step).pack(side='left')
    Button(buttonframe, text='Shift', underline=0, background='#90f090', foreground='black', command=self.shift).pack(side='left')
    Button(buttonframe, text='Reduce', underline=0, background='#90f090', foreground='black', command=self.reduce).pack(side='left')
    Button(buttonframe, text='Undo', underline=0, background='#f0a0a0', foreground='black', command=self.undo).pack(side='left')