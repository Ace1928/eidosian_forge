import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def _discourses_block_reader(self, stream):
    return [[self._parse_utterance(u, include_tag=False) for b in read_blankline_block(stream) for u in b.split('\n') if u.strip()]]