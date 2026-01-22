import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
@total_ordering
class CustomFeatureValue:
    """
    An abstract base class for base values that define a custom
    unification method.  The custom unification method of
    ``CustomFeatureValue`` will be used during unification if:

      - The ``CustomFeatureValue`` is unified with another base value.
      - The ``CustomFeatureValue`` is not the value of a customized
        ``Feature`` (which defines its own unification method).

    If two ``CustomFeatureValue`` objects are unified with one another
    during feature structure unification, then the unified base values
    they return *must* be equal; otherwise, an ``AssertionError`` will
    be raised.

    Subclasses must define ``unify()``, ``__eq__()`` and ``__lt__()``.
    Subclasses may also wish to define ``__hash__()``.
    """

    def unify(self, other):
        """
        If this base value unifies with ``other``, then return the
        unified value.  Otherwise, return ``UnificationFailure``.
        """
        raise NotImplementedError('abstract base class')

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return NotImplemented

    def __hash__(self):
        raise TypeError('%s objects or unhashable' % self.__class__.__name__)