import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class unit_tokenize(tokenize):
    """Tokenizer class that yields the text as a single token."""
    _DOC_ERRORS = []

    def __init__(self, text):
        super().__init__(text)
        self._done = False

    def next(self):
        if self._done:
            raise StopIteration()
        self._done = True
        return (self._text, 0)