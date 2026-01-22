import types
from ._impl import (
class AllMatch:
    """Matches if all provided values match the given matcher."""

    def __init__(self, matcher):
        self.matcher = matcher

    def __str__(self):
        return f'AllMatch({self.matcher})'

    def match(self, values):
        mismatches = []
        for value in values:
            mismatch = self.matcher.match(value)
            if mismatch:
                mismatches.append(mismatch)
        if mismatches:
            return MismatchesAll(mismatches)