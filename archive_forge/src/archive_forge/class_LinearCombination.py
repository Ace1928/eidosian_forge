from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
class LinearCombination(object):
    """
    Linear combination of Variable differentials.

    The linear_combo attribute can change formally, but its value
    always remains the same. Typically, the linear combination can
    thus be expanded.

    The expanded form of linear_combo is a mapping from Variables to
    the coefficient of their differential.
    """
    __slots__ = 'linear_combo'

    def __init__(self, linear_combo):
        """
        linear_combo can be modified by the object, during its
        lifetime. This allows the object to change its internal
        representation over time (for instance by expanding the linear
        combination and replacing the original expression with the
        expanded one).

        linear_combo -- if linear_combo is a dict, then it represents
        an expanded linear combination and must map Variables to the
        coefficient of their differential. Otherwise, it should be a
        list of (coefficient, LinearCombination) pairs (that
        represents a linear combination expression).
        """
        self.linear_combo = linear_combo

    def __bool__(self):
        """
        Return True only if the linear combination is non-empty, i.e. if
        the linear combination contains any term.
        """
        return bool(self.linear_combo)

    def expanded(self):
        """
        Return True if and only if the linear combination is expanded.
        """
        return isinstance(self.linear_combo, dict)

    def expand(self):
        """
        Expand the linear combination.

        The expansion is a collections.defaultdict(float).

        This should only be called if the linear combination is not
        yet expanded.
        """
        derivatives = collections.defaultdict(float)
        while self.linear_combo:
            main_factor, main_expr = self.linear_combo.pop()
            if main_expr.expanded():
                for var, factor in main_expr.linear_combo.items():
                    derivatives[var] += main_factor * factor
            else:
                for factor, expr in main_expr.linear_combo:
                    self.linear_combo.append((main_factor * factor, expr))
        self.linear_combo = derivatives

    def __getstate__(self):
        return (self.linear_combo,)

    def __setstate__(self, state):
        self.linear_combo, = state