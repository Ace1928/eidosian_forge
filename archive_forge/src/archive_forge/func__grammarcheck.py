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
def _grammarcheck(self, grammar):
    if self.top is None:
        return
    self.grammarbox.tag_remove('error', '1.0', 'end')
    self._grammarcheck_errs = []
    for lineno, line in enumerate(grammar.split('\n')):
        line = re.sub('((\\\\.|[^#])*)(#.*)?', '\\1', line)
        line = line.strip()
        if line:
            try:
                RegexpChunkRule.fromstring(line)
            except ValueError as e:
                self.grammarbox.tag_add('error', '%s.0' % (lineno + 1), '%s.0 lineend' % (lineno + 1))
    self.status['text'] = ''