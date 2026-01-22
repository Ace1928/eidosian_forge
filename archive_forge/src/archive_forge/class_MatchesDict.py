from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class MatchesDict(_CombinedMatcher):
    """Match a dictionary exactly, by its keys.

    Specify a dictionary mapping keys (often strings) to matchers.  This is
    the 'expected' dict.  Any dictionary that matches this must have exactly
    the same keys, and the values must match the corresponding matchers in the
    expected dict.
    """
    matcher_factories = {'Extra': _SubDictOf, 'Missing': lambda m: _SuperDictOf(m, format_value=str), 'Differences': _MatchCommonKeys}
    format_expected = lambda self, expected: _format_matcher_dict(expected)