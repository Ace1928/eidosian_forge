import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def assertToken(self, tok, expected):
    if isinstance(expected, list):
        if tok not in expected:
            raise UnexpectedTokenException(self._currentIndex, tok, expected)
    elif tok != expected:
        raise UnexpectedTokenException(self._currentIndex, tok, expected)