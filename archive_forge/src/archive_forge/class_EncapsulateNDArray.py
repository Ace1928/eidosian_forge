from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
class EncapsulateNDArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A class that "mocks" ndarray by encapsulating an ndarray and using
    protocols to "look like" an ndarray. Basically tests whether Dask
    works fine with something that is essentially an array but uses
    protocols instead of being an actual array. Must be manually
    registered as a valid chunk type to be considered a downcast type
    of Dask array in the type casting hierarchy.
    """
    __array_priority__ = 20

    def __init__(self, arr):
        self.arr = arr

    def __array__(self, *args, **kwargs):
        return np.asarray(self.arr, *args, **kwargs)

    def __array_function__(self, f, t, arrs, kw):
        if not all((issubclass(ti, (type(self), np.ndarray) + np.ScalarType) for ti in t)):
            return NotImplemented
        arrs = tuple((arr if not isinstance(arr, type(self)) else arr.arr for arr in arrs))
        t = tuple((ti for ti in t if not issubclass(ti, type(self))))
        print(t)
        a = self.arr.__array_function__(f, t, arrs, kw)
        return a if not isinstance(a, np.ndarray) else type(self)(a)
    __getitem__ = wrap('__getitem__')
    __setitem__ = wrap('__setitem__')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not all((isinstance(i, (type(self), np.ndarray) + np.ScalarType) for i in inputs)):
            return NotImplemented
        inputs = tuple((i if not isinstance(i, type(self)) else i.arr for i in inputs))
        a = getattr(ufunc, method)(*inputs, **kwargs)
        return a if not isinstance(a, np.ndarray) else type(self)(a)
    shape = dispatch_property('shape')
    ndim = dispatch_property('ndim')
    dtype = dispatch_property('dtype')
    astype = wrap('astype')
    sum = wrap('sum')
    prod = wrap('prod')
    reshape = wrap('reshape')
    squeeze = wrap('squeeze')