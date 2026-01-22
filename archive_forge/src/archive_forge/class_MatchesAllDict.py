from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class MatchesAllDict(Matcher):
    """Matches if all of the matchers it is created with match.

    A lot like ``MatchesAll``, but takes a dict of Matchers and labels any
    mismatches with the key of the dictionary.
    """

    def __init__(self, matchers):
        super().__init__()
        self.matchers = matchers

    def __str__(self):
        return f'MatchesAllDict({_format_matcher_dict(self.matchers)})'

    def match(self, observed):
        mismatches = {}
        for label in self.matchers:
            mismatches[label] = self.matchers[label].match(observed)
        return _dict_to_mismatch(mismatches, result_mismatch=LabelledMismatches)