from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
class PydapArrayWrapper(BackendArray):

    def __init__(self, array):
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, key):
        array = getattr(self.array, 'array', self.array)
        result = robust_getitem(array, key, catch=ValueError)
        result = np.asarray(result)
        axis = tuple((n for n, k in enumerate(key) if isinstance(k, integer_types)))
        if result.ndim + len(axis) != array.ndim and axis:
            result = np.squeeze(result, axis)
        return result