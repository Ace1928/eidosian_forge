from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
def _take_without_fill(self, indices) -> Self:
    to_shift = indices < 0
    n = len(self)
    if indices.max() >= n or indices.min() < -n:
        if n == 0:
            raise IndexError('cannot do a non-empty take from an empty axes.')
        raise IndexError("out of bounds value in 'indices'.")
    if to_shift.any():
        indices = indices.copy()
        indices[to_shift] += n
    sp_indexer = self.sp_index.lookup_array(indices)
    value_mask = sp_indexer != -1
    new_sp_values = self.sp_values[sp_indexer[value_mask]]
    value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)
    new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
    return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)