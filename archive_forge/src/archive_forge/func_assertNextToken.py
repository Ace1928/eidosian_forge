import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def assertNextToken(self, expected):
    try:
        tok = self.token()
    except ExpectedMoreTokensException as e:
        raise ExpectedMoreTokensException(e.index, message="Expected token '%s'." % expected) from e
    if isinstance(expected, list):
        if tok not in expected:
            raise UnexpectedTokenException(self._currentIndex, tok, expected)
    elif tok != expected:
        raise UnexpectedTokenException(self._currentIndex, tok, expected)