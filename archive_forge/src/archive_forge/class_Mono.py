from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
class Mono(metaclass=Type):
    """
    Monotype are unqualified 0 parameters.

    Each type must be reconstructable using its parameters:

        type(datashape_type)(*type.parameters)
    """
    composite = False

    def __init__(self, *params):
        self._parameters = params

    @property
    def _slotted(self):
        return hasattr(self, '__slots__')

    @property
    def parameters(self):
        if self._slotted:
            return tuple((getattr(self, slot) for slot in self.__slots__))
        else:
            return self._parameters

    def info(self):
        return (type(self), self.parameters)

    def __eq__(self, other):
        return isinstance(other, Mono) and self.shape == other.shape and (self.measure.info() == other.measure.info())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        try:
            h = self._hash
        except AttributeError:
            h = self._hash = hash(self.shape) ^ hash(self.measure.info())
        return h

    @property
    def shape(self):
        return ()

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return [self][key]

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join(('%s=%r' % (slot, getattr(self, slot)) for slot in self.__slots__) if self._slotted else map(repr, self.parameters)))

    @property
    def measure(self):
        return self

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed. In the case of a measure such as CType,
        'leading' must be 0, and self is returned.
        """
        if leading >= 1:
            raise IndexError('Not enough dimensions in data shape to remove %d leading dimensions.' % leading)
        else:
            return self

    def __mul__(self, other):
        if isinstance(other, str):
            from datashader import datashape
            return datashape.dshape(other).__rmul__(self)
        if isinstance(other, int):
            other = Fixed(other)
        if isinstance(other, DataShape):
            return other.__rmul__(self)
        return DataShape(self, other)

    def __rmul__(self, other):
        if isinstance(other, str):
            from datashader import datashape
            return self * datashape.dshape(other)
        if isinstance(other, int):
            other = Fixed(other)
        return DataShape(other, self)

    def __getstate__(self):
        return self.parameters

    def __setstate__(self, state):
        if self._slotted:
            for slot, val in zip(self.__slots__, state):
                setattr(self, slot, val)
        else:
            self._parameters = state

    def to_numpy_dtype(self):
        raise TypeError('DataShape %s is not NumPy-compatible' % self)