import random
import re
import textwrap
import time
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle
def _view_history(self, index):
    index = max(0, min(len(self._history) - 1, index))
    if not self._history:
        return
    if index == self._history_index:
        return
    self.grammarbox['state'] = 'normal'
    self.grammarbox.delete('1.0', 'end')
    self.grammarbox.insert('end', self._history[index][0])
    self.grammarbox.mark_set('insert', '1.0')
    self._history_index = index
    self._syntax_highlight_grammar(self._history[index][0])
    self.normalized_grammar = self.normalize_grammar(self._history[index][0])
    if self.normalized_grammar:
        rules = [RegexpChunkRule.fromstring(line) for line in self.normalized_grammar.split('\n')]
    else:
        rules = []
    self.chunker = RegexpChunkParser(rules)
    self._eval_plot()
    self._highlight_devset()
    if self._showing_trace:
        self.show_trace()
    if self._history_index < len(self._history) - 1:
        self.grammarlabel['text'] = 'Grammar {}/{}:'.format(self._history_index + 1, len(self._history))
    else:
        self.grammarlabel['text'] = 'Grammar:'