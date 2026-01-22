import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def _words_block_reader(self, stream):
    return sum(self._discourses_block_reader(stream)[0], [])