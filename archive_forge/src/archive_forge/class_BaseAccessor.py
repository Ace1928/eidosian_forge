from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype
from pandas.core.accessor import (
from pandas.core.arrays.sparse.array import SparseArray
class BaseAccessor:
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None) -> None:
        self._parent = data
        self._validate(data)

    def _validate(self, data):
        raise NotImplementedError