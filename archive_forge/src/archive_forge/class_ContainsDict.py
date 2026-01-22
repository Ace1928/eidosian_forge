from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class ContainsDict(_CombinedMatcher):
    """Match a dictionary for that contains a specified sub-dictionary.

    Specify a dictionary mapping keys (often strings) to matchers.  This is
    the 'expected' dict.  Any dictionary that matches this must have **at
    least** these keys, and the values must match the corresponding matchers
    in the expected dict.  Dictionaries that have more keys will also match.

    In other words, any matching dictionary must contain the dictionary given
    to the constructor.

    Does not check for strict sub-dictionary.  That is, equal dictionaries
    match.
    """
    matcher_factories = {'Missing': lambda m: _SuperDictOf(m, format_value=str), 'Differences': _MatchCommonKeys}
    format_expected = lambda self, expected: _format_matcher_dict(expected)