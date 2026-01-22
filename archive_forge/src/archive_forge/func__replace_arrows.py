import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _replace_arrows(self, *e):
    """
        Replace any ``'->'`` text strings with arrows (char \\256, in
        symbol font).  This searches the whole buffer, but is fast
        enough to be done anytime they press '>'.
        """
    arrow = '1.0'
    while True:
        arrow = self._textwidget.search('->', arrow, 'end+1char')
        if arrow == '':
            break
        self._textwidget.delete(arrow, arrow + '+2char')
        self._textwidget.insert(arrow, self.ARROW, 'arrow')
        self._textwidget.insert(arrow, '\t')
    arrow = '1.0'
    while True:
        arrow = self._textwidget.search(self.ARROW, arrow + '+1char', 'end+1char')
        if arrow == '':
            break
        self._textwidget.tag_add('arrow', arrow, arrow + '+1char')