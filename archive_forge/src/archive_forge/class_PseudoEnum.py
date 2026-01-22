import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
class PseudoEnum:
    """A base class for types which resemble enumeration types."""

    def __init__(self, name, order):
        self._name = name
        self._order = order

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._name)

    def __str__(self):
        return self._name

    def __lt__(self, other):
        return self._order < other._order

    def __le__(self, other):
        return self._order <= other._order

    def __eq__(self, other):
        return self._order == other._order

    def __ne__(self, other):
        return self._order != other._order

    def __ge__(self, other):
        return self._order >= other._order

    def __gt__(self, other):
        return self._order > other._order

    def __hash__(self):
        return hash(self._order)