from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
class MarkdownSection(MarkdownBlock):

    def __init__(self, heading, level, *args):
        self.heading = heading
        self.level = level
        super().__init__(*args)