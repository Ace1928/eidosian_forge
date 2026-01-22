import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _init_nonterminal_tag(self, tag, foreground='blue'):
    self._textwidget.tag_config(tag, foreground=foreground, font=CFGEditor._BOLD)
    if not self._highlight_matching_nonterminals:
        return

    def enter(e, textwidget=self._textwidget, tag=tag):
        textwidget.tag_config(tag, background='#80ff80')

    def leave(e, textwidget=self._textwidget, tag=tag):
        textwidget.tag_config(tag, background='')
    self._textwidget.tag_bind(tag, '<Enter>', enter)
    self._textwidget.tag_bind(tag, '<Leave>', leave)