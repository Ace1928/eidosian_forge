from __future__ import annotations
import math
import numpy as np
from dask.array import chunk
from dask.array.core import Array
from dask.array.dispatch import (
from dask.array.numpy_compat import divide as np_divide
from dask.array.numpy_compat import ma_divide
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
class ArrayBackendEntrypoint(DaskBackendEntrypoint):
    """Dask-Array version of ``DaskBackendEntrypoint``

    See Also
    --------
    NumpyBackendEntrypoint
    """

    @property
    def RandomState(self):
        """Return the backend-specific RandomState class

        For example, the 'numpy' backend simply returns
        ``numpy.random.RandomState``.
        """
        raise NotImplementedError

    @property
    def default_bit_generator(self):
        """Return the default BitGenerator type"""
        raise NotImplementedError

    @staticmethod
    def ones(shape, *, dtype=None, meta=None, **kwargs):
        """Create an array of ones

        Returns a new array having a specified shape and filled
        with ones.
        """
        raise NotImplementedError

    @staticmethod
    def zeros(shape, *, dtype=None, meta=None, **kwargs):
        """Create an array of zeros

        Returns a new array having a specified shape and filled
        with zeros.
        """
        raise NotImplementedError

    @staticmethod
    def empty(shape, *, dtype=None, meta=None, **kwargs):
        """Create an empty array

        Returns an uninitialized array having a specified shape.
        """
        raise NotImplementedError

    @staticmethod
    def full(shape, fill_value, *, dtype=None, meta=None, **kwargs):
        """Create a uniformly filled array

        Returns a new array having a specified shape and filled
        with fill_value.
        """
        raise NotImplementedError

    @staticmethod
    def arange(start, /, stop=None, step=1, *, dtype=None, meta=None, **kwargs):
        """Create an ascending or descending array

        Returns evenly spaced values within the half-open interval
        ``[start, stop)`` as a one-dimensional array.
        """
        raise NotImplementedError