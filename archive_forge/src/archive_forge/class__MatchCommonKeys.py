from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class _MatchCommonKeys(Matcher):
    """Match on keys in a dictionary.

    Given a dictionary where the values are matchers, this will look for
    common keys in the matched dictionary and match if and only if all common
    keys match the given matchers.

    Thus::

      >>> structure = {'a': Equals('x'), 'b': Equals('y')}
      >>> _MatchCommonKeys(structure).match({'a': 'x', 'c': 'z'})
      None
    """

    def __init__(self, dict_of_matchers):
        super().__init__()
        self._matchers = dict_of_matchers

    def _compare_dicts(self, expected, observed):
        common_keys = set(expected.keys()) & set(observed.keys())
        mismatches = {}
        for key in common_keys:
            mismatch = expected[key].match(observed[key])
            if mismatch:
                mismatches[key] = mismatch
        return mismatches

    def match(self, observed):
        mismatches = self._compare_dicts(self._matchers, observed)
        if mismatches:
            return DictMismatches(mismatches)