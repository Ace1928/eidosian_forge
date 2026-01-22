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
class NumpyBackendEntrypoint(ArrayBackendEntrypoint):

    @classmethod
    def to_backend_dispatch(cls):
        return to_numpy_dispatch

    @classmethod
    def to_backend(cls, data: Array, **kwargs):
        if isinstance(data._meta, np.ndarray):
            return data
        return data.map_blocks(cls.to_backend_dispatch(), **kwargs)

    @property
    def RandomState(self):
        return np.random.RandomState

    @property
    def default_bit_generator(self):
        return np.random.PCG64