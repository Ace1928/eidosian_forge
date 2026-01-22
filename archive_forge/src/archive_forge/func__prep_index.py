from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype
from pandas.core.accessor import (
from pandas.core.arrays.sparse.array import SparseArray
@staticmethod
def _prep_index(data, index, columns):
    from pandas.core.indexes.api import default_index, ensure_index
    N, K = data.shape
    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)
    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    if len(columns) != K:
        raise ValueError(f'Column length mismatch: {len(columns)} vs. {K}')
    if len(index) != N:
        raise ValueError(f'Index length mismatch: {len(index)} vs. {N}')
    return (index, columns)