from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
class MutableSliceableSequence(MutableSequence):
    __slots__ = ()

    def __getitem__(self, index):
        if not isinstance(index, slice):
            return self.__getsingleitem__(index)
        return type(self)([self[i] for i in range(*index.indices(len(self)))])

    def __setitem__(self, index, value):
        if not isinstance(index, slice):
            return self.__setsingleitem__(index, value)
        assert iter(value)
        if index.step is None:
            del self[index.start:index.stop]
            for elem in reversed(value):
                self.insert(0 if index.start is None else index.start, elem)
        else:
            range_parms = index.indices(len(self))
            nr_assigned_items = (range_parms[1] - range_parms[0] - 1) // range_parms[2] + 1
            if nr_assigned_items < len(value):
                raise TypeError('too many elements in value {} < {}'.format(nr_assigned_items, len(value)))
            elif nr_assigned_items > len(value):
                raise TypeError('not enough elements in value {} > {}'.format(nr_assigned_items, len(value)))
            for idx, i in enumerate(range(*range_parms)):
                self[i] = value[idx]

    def __delitem__(self, index):
        if not isinstance(index, slice):
            return self.__delsingleitem__(index)
        for i in reversed(range(*index.indices(len(self)))):
            del self[i]

    @abstractmethod
    def __getsingleitem__(self, index):
        raise IndexError

    @abstractmethod
    def __setsingleitem__(self, index, value):
        raise IndexError

    @abstractmethod
    def __delsingleitem__(self, index):
        raise IndexError