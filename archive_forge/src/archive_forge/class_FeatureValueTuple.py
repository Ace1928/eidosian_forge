import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatureValueTuple(SubstituteBindingsSequence, tuple):
    """
    A base feature value that is a tuple of other base feature values.
    FeatureValueTuple implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueTuple`` is immutable.
    """

    def __repr__(self):
        if len(self) == 0:
            return '()'
        return '(%s)' % ', '.join((f'{b}' for b in self))