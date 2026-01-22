import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class UnknownPosition(Position):
    """Specify a specific position which is unknown (has no position).

    This is used in UniProt, e.g. ? or in the XML as unknown.
    """

    def __repr__(self):
        """Represent the UnknownPosition object as a string for debugging."""
        return f'{self.__class__.__name__}()'

    def __hash__(self):
        """Return the hash value of the UnknownPosition object."""
        return hash(None)

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted (PRIVATE)."""
        return self

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return self