from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_grammar(self, parent):
    self._prodframe = listframe = Frame(parent)
    self._prodframe.pack(fill='both', side='left', padx=2)
    self._prodlist_label = Label(self._prodframe, font=self._boldfont, text='Available Reductions')
    self._prodlist_label.pack()
    self._prodlist = Listbox(self._prodframe, selectmode='single', relief='groove', background='white', foreground='#909090', font=self._font, selectforeground='#004040', selectbackground='#c0f0c0')
    self._prodlist.pack(side='right', fill='both', expand=1)
    self._productions = list(self._parser.grammar().productions())
    for production in self._productions:
        self._prodlist.insert('end', ' %s' % production)
    self._prodlist.config(height=min(len(self._productions), 25))
    if 1:
        listscroll = Scrollbar(self._prodframe, orient='vertical')
        self._prodlist.config(yscrollcommand=listscroll.set)
        listscroll.config(command=self._prodlist.yview)
        listscroll.pack(side='left', fill='y')
    self._prodlist.bind('<<ListboxSelect>>', self._prodlist_select)
    self._hover = -1
    self._prodlist.bind('<Motion>', self._highlight_hover)
    self._prodlist.bind('<Leave>', self._clear_hover)