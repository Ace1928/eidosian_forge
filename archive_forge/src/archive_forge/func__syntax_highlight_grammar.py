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
def _syntax_highlight_grammar(self, grammar):
    if self.top is None:
        return
    self.grammarbox.tag_remove('comment', '1.0', 'end')
    self.grammarbox.tag_remove('angle', '1.0', 'end')
    self.grammarbox.tag_remove('brace', '1.0', 'end')
    self.grammarbox.tag_add('hangindent', '1.0', 'end')
    for lineno, line in enumerate(grammar.split('\n')):
        if not line.strip():
            continue
        m = re.match('(\\\\.|[^#])*(#.*)?', line)
        comment_start = None
        if m.group(2):
            comment_start = m.start(2)
            s = '%d.%d' % (lineno + 1, m.start(2))
            e = '%d.%d' % (lineno + 1, m.end(2))
            self.grammarbox.tag_add('comment', s, e)
        for m in re.finditer('[<>{}]', line):
            if comment_start is not None and m.start() >= comment_start:
                break
            s = '%d.%d' % (lineno + 1, m.start())
            e = '%d.%d' % (lineno + 1, m.end())
            if m.group() in '<>':
                self.grammarbox.tag_add('angle', s, e)
            else:
                self.grammarbox.tag_add('brace', s, e)