import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class SameMembers(Matcher):
    """Matches if two iterators have the same members.

    This is not the same as set equivalence.  The two iterators must be of the
    same length and have the same repetitions.
    """

    def __init__(self, expected):
        super().__init__()
        self.expected = expected

    def __str__(self):
        return f'{self.__class__.__name__}({self.expected!r})'

    def match(self, observed):
        expected_only = list_subtract(self.expected, observed)
        observed_only = list_subtract(observed, self.expected)
        if expected_only == observed_only == []:
            return
        return PostfixedMismatch('\nmissing:    {}\nextra:      {}'.format(_format(expected_only), _format(observed_only)), _BinaryMismatch(observed, 'elements differ', self.expected))