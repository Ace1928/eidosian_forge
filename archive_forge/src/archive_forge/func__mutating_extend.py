from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _mutating_extend(self, sequence):
    offset = 0
    sequence_len = len(sequence)
    while offset < sequence_len:
        offset = self._mutating_fill_tail(offset, sequence)
        if len(self._tail) == BRANCH_FACTOR:
            self._mutating_insert_tail()
    self._tail_offset = self._count - len(self._tail)