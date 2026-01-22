import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class _BinaryComparison:
    """Matcher that compares an object to another object."""

    def __init__(self, expected):
        self.expected = expected

    def __str__(self):
        return f'{self.__class__.__name__}({self.expected!r})'

    def match(self, other):
        if self.comparator(other, self.expected):
            return None
        return _BinaryMismatch(other, self.mismatch_string, self.expected)

    def comparator(self, expected, other):
        raise NotImplementedError(self.comparator)