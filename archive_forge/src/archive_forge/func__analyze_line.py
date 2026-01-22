import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _analyze_line(self, linenum):
    """
        Colorize a given line.
        """
    self._clear_tags(linenum)
    line = self._textwidget.get(repr(linenum) + '.0', repr(linenum) + '.end')
    if CFGEditor._PRODUCTION_RE.match(line):

        def analyze_token(match, self=self, linenum=linenum):
            self._analyze_token(match, linenum)
            return ''
        CFGEditor._TOKEN_RE.sub(analyze_token, line)
    elif line.strip() != '':
        self._mark_error(linenum, line)