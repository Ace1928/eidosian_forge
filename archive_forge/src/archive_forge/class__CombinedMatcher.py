from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class _CombinedMatcher(Matcher):
    """Many matchers labelled and combined into one uber-matcher.

    Subclass this and then specify a dict of matcher factories that take a
    single 'expected' value and return a matcher.  The subclass will match
    only if all of the matchers made from factories match.

    Not **entirely** dissimilar from ``MatchesAll``.
    """
    matcher_factories = {}

    def __init__(self, expected):
        super().__init__()
        self._expected = expected

    def format_expected(self, expected):
        return repr(expected)

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.format_expected(self._expected))

    def match(self, observed):
        matchers = {k: v(self._expected) for k, v in self.matcher_factories.items()}
        return MatchesAllDict(matchers).match(observed)