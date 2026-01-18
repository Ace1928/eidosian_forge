from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
from nltk.internals import overridden
from nltk.tokenize.util import string_span_tokenize
def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
    """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: List[List[str]]
        """
    return [self.tokenize(s) for s in strings]