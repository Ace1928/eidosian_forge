from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
class MatchesListwise:
    """Matches if each matcher matches the corresponding value.

    More easily explained by example than in words:

    >>> from ._basic import Equals
    >>> MatchesListwise([Equals(1)]).match([1])
    >>> MatchesListwise([Equals(1), Equals(2)]).match([1, 2])
    >>> print (MatchesListwise([Equals(1), Equals(2)]).match([2, 1]).describe())
    Differences: [
    2 != 1
    1 != 2
    ]
    >>> matcher = MatchesListwise([Equals(1), Equals(2)], first_only=True)
    >>> print (matcher.match([3, 4]).describe())
    3 != 1
    """

    def __init__(self, matchers, first_only=False):
        """Construct a MatchesListwise matcher.

        :param matchers: A list of matcher that the matched values must match.
        :param first_only: If True, then only report the first mismatch,
            otherwise report all of them. Defaults to False.
        """
        self.matchers = matchers
        self.first_only = first_only

    def match(self, values):
        from ._basic import HasLength
        mismatches = []
        length_mismatch = Annotate('Length mismatch', HasLength(len(self.matchers))).match(values)
        if length_mismatch:
            mismatches.append(length_mismatch)
        for matcher, value in zip(self.matchers, values):
            mismatch = matcher.match(value)
            if mismatch:
                if self.first_only:
                    return mismatch
                mismatches.append(mismatch)
        if mismatches:
            return MismatchesAll(mismatches)