import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class OneOfPosition(int, Position):
    """Specify a position where the location can be multiple positions.

    This models the GenBank 'one-of(1888,1901)' function, and tries
    to make this fit within the Biopython Position models. If this was
    a start position it should act like 1888, but as an end position 1901.

    >>> p = OneOfPosition(1888, [ExactPosition(1888), ExactPosition(1901)])
    >>> p
    OneOfPosition(1888, choices=[ExactPosition(1888), ExactPosition(1901)])
    >>> int(p)
    1888

    Integer comparisons and operators act like using int(p),

    >>> p == 1888
    True
    >>> p <= 1888
    True
    >>> p > 1888
    False
    >>> p + 100
    OneOfPosition(1988, choices=[ExactPosition(1988), ExactPosition(2001)])

    >>> isinstance(p, OneOfPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    """

    def __new__(cls, position, choices):
        """Initialize with a set of possible positions.

        choices is a list of Position derived objects, specifying possible
        locations.

        position is an integer specifying the default behavior.
        """
        if position not in choices:
            raise ValueError(f'OneOfPosition: {position!r} should match one of {choices!r}')
        obj = int.__new__(cls, position)
        obj.position_choices = choices
        return obj

    def __getnewargs__(self):
        """Return the arguments accepted by __new__.

        Necessary to allow pickling and unpickling of class instances.
        """
        return (int(self), self.position_choices)

    def __repr__(self):
        """Represent the OneOfPosition object as a string for debugging."""
        return '%s(%i, choices=%r)' % (self.__class__.__name__, int(self), self.position_choices)

    def __str__(self):
        """Return a representation of the OneOfPosition object (with python counting)."""
        out = 'one-of('
        for position in self.position_choices:
            out += f'{position},'
        return out[:-1] + ')'

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted (PRIVATE)."""
        return self.__class__(int(self) + offset, [p + offset for p in self.position_choices])

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return self.__class__(length - int(self), [p._flip(length) for p in self.position_choices[::-1]])