import re
import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation
def _elt_to_words(self, elt, handler):
    return [self._simplify_username(t.attrib['word']) for t in elt.findall('t')]