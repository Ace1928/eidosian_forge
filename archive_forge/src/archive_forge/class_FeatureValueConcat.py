import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatureValueConcat(SubstituteBindingsSequence, tuple):
    """
    A base feature value that represents the concatenation of two or
    more ``FeatureValueTuple`` or ``Variable``.
    """

    def __new__(cls, values):
        values = _flatten(values, FeatureValueConcat)
        if sum((isinstance(v, Variable) for v in values)) == 0:
            values = _flatten(values, FeatureValueTuple)
            return FeatureValueTuple(values)
        if len(values) == 1:
            return list(values)[0]
        return tuple.__new__(cls, values)

    def __repr__(self):
        return '(%s)' % '+'.join((f'{b}' for b in self))