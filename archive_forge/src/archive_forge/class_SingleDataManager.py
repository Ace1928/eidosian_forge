from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import (
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
class SingleDataManager(DataManager):

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @final
    @property
    def array(self) -> ArrayLike:
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
        return self.arrays[0]

    def setitem_inplace(self, indexer, value, warn: bool=True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        arr = self.array
        if isinstance(arr, np.ndarray):
            value = np_can_hold_element(arr.dtype, value)
        if isinstance(value, np.ndarray) and value.ndim == 1 and (len(value) == 1):
            value = value[0, ...]
        arr[indexer] = value

    def grouped_reduce(self, func):
        arr = self.array
        res = func(arr)
        index = default_index(len(res))
        mgr = type(self).from_array(res, index)
        return mgr

    @classmethod
    def from_array(cls, arr: ArrayLike, index: Index):
        raise AbstractMethodError(cls)