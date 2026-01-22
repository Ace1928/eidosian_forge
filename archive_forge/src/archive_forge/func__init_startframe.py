import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _init_startframe(self):
    frame = self._startframe = Frame(self._top)
    self._start = Entry(frame)
    self._start.pack(side='right')
    Label(frame, text='Start Symbol:').pack(side='right')
    Label(frame, text='Productions:').pack(side='left')
    self._start.insert(0, self._cfg.start().symbol())