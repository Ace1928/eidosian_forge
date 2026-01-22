import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class AfterPosition(int, Position):
    """Specify a position where the actual location is found after it.

    Arguments:
     - position - The lower boundary of where the location can occur.
     - extension - An optional argument which must be zero since we don't
       have an extension. The argument is provided so that the same number
       of arguments can be passed to all position types.

    This is used to specify positions like (>10..100) where the location
    occurs somewhere after position 10.

    >>> p = AfterPosition(7)
    >>> p
    AfterPosition(7)
    >>> print(p)
    >7
    >>> int(p)
    7
    >>> p + 10
    AfterPosition(17)

    >>> isinstance(p, AfterPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Note this potentially surprising behavior:

    >>> p == ExactPosition(7)
    True
    >>> p == BeforePosition(7)
    True

    Just remember that for equality and sorting the position objects act
    like integers.
    """

    def __new__(cls, position, extension=0):
        """Create a new instance of the AfterPosition object."""
        if extension != 0:
            raise AttributeError(f'Non-zero extension {extension} for exact position.')
        return int.__new__(cls, position)

    def __repr__(self):
        """Represent the location as a string for debugging."""
        return '%s(%i)' % (self.__class__.__name__, int(self))

    def __str__(self):
        """Return a representation of the AfterPosition object (with python counting)."""
        return f'>{int(self)}'

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted (PRIVATE)."""
        return self.__class__(int(self) + offset)

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return BeforePosition(length - int(self))