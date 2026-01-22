from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
from nltk.internals import overridden
from nltk.tokenize.util import string_span_tokenize
class StringTokenizer(TokenizerI):
    """A tokenizer that divides a string into substrings by splitting
    on the specified string (defined in subclasses).
    """

    @property
    @abstractmethod
    def _string(self):
        raise NotImplementedError

    def tokenize(self, s):
        return s.split(self._string)

    def span_tokenize(self, s):
        yield from string_span_tokenize(s, self._string)