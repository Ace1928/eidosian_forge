import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
class _UndefinedSequenceData(SequenceDataAbstractBaseClass):
    """Stores the length of a sequence with an undefined sequence contents (PRIVATE).

    Objects of this class can be used to create a Seq object to represent
    sequences with a known length, but an unknown sequence contents.
    Calling __len__ returns the sequence length, calling __getitem__ raises an
    UndefinedSequenceError except for requests of zero size, for which it
    returns an empty bytes object.
    """
    __slots__ = ('_length',)

    def __init__(self, length):
        """Initialize the object with the sequence length.

        The calling function is responsible for ensuring that the length is
        greater than zero.
        """
        self._length = length
        super().__init__()

    def __getitem__(self, key: slice) -> Union[bytes, '_UndefinedSequenceData']:
        if isinstance(key, slice):
            start, end, step = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
            return _UndefinedSequenceData(size)
        else:
            raise UndefinedSequenceError('Sequence content is undefined')

    def __len__(self):
        return self._length

    def __bytes__(self):
        raise UndefinedSequenceError('Sequence content is undefined')

    def __add__(self, other):
        length = len(self) + len(other)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            if isinstance(other, _UndefinedSequenceData):
                return _UndefinedSequenceData(length)
            else:
                return NotImplemented
        else:
            data = {len(self): other}
            return _PartiallyDefinedSequenceData(length, data)

    def __radd__(self, other):
        data = {0: bytes(other)}
        length = len(other) + len(self)
        return _PartiallyDefinedSequenceData(length, data)

    def upper(self):
        """Return an upper case copy of the sequence."""
        return _UndefinedSequenceData(self._length)

    def lower(self):
        """Return a lower case copy of the sequence."""
        return _UndefinedSequenceData(self._length)

    def isupper(self):
        """Return True if all ASCII characters in data are uppercase.

        If there are no cased characters, the method returns False.
        """
        raise UndefinedSequenceError('Sequence content is undefined')

    def islower(self):
        """Return True if all ASCII characters in data are lowercase.

        If there are no cased characters, the method returns False.
        """
        raise UndefinedSequenceError('Sequence content is undefined')

    def replace(self, old, new):
        """Return a copy with all occurrences of substring old replaced by new."""
        if len(old) != len(new):
            raise UndefinedSequenceError('Sequence content is undefined')
        return _UndefinedSequenceData(self._length)

    @property
    def defined(self):
        """Return False, as the sequence is not defined and has a non-zero length."""
        return False

    @property
    def defined_ranges(self):
        """Return a tuple of the ranges where the sequence contents is defined.

        As the sequence contents of an _UndefinedSequenceData object is fully
        undefined, the return value is always an empty tuple.
        """
        return ()