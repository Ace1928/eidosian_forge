import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
class AnyRange(object):
    """A range object for representing Any sets"""
    __slots__ = ()

    def __init__(self):
        pass

    def __str__(self):
        return '[*]'
    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, AnyRange)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        return True

    def isdiscrete(self):
        return False

    def isfinite(self):
        return False

    def isdisjoint(self, other):
        return False

    def issubset(self, other):
        return isinstance(other, AnyRange)

    def range_difference(self, other_ranges):
        for r in other_ranges:
            if isinstance(r, AnyRange):
                return []
        else:
            return [self]

    def range_intersection(self, other_ranges):
        return list(other_ranges)