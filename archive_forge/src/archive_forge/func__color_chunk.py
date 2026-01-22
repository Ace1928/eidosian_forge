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
def _color_chunk(self, sentnum, chunk, tag):
    start, end = chunk
    self.devsetbox.tag_add(tag, f'{self.linenum[sentnum]}.{self.charnum[sentnum, start]}', f'{self.linenum[sentnum]}.{self.charnum[sentnum, end] - 1}')