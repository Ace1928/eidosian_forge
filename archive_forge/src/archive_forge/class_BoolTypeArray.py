from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class BoolTypeArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Decode arrays on the fly from integer to boolean datatype

    This is useful for decoding boolean arrays from integer typed netCDF
    variables.

    >>> x = np.array([1, 0, 1, 1, 0], dtype="i1")

    >>> x.dtype
    dtype('int8')

    >>> BoolTypeArray(x).dtype
    dtype('bool')

    >>> indexer = indexing.BasicIndexer((slice(None),))
    >>> BoolTypeArray(x)[indexer].dtype
    dtype('bool')
    """
    __slots__ = ('array',)

    def __init__(self, array) -> None:
        self.array = indexing.as_indexable(array)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype('bool')

    def _oindex_get(self, key):
        return np.asarray(self.array.oindex[key], dtype=self.dtype)

    def _vindex_get(self, key):
        return np.asarray(self.array.vindex[key], dtype=self.dtype)

    def __getitem__(self, key) -> np.ndarray:
        return np.asarray(self.array[key], dtype=self.dtype)