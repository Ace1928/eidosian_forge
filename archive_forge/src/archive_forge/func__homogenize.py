from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _homogenize(data, index: Index, dtype: DtypeObj | None) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    refs: list[Any] = []
    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype, copy=False)
            if isinstance(val, ABCSeries) and val.index is not index:
                val = val.reindex(index, copy=False)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
                if oindex is None:
                    oindex = index.astype('O')
                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    val = dict_compat(val)
                else:
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)
            val = sanitize_array(val, index, dtype=dtype, copy=False)
            com.require_length_match(val, index)
            refs.append(None)
        homogenized.append(val)
    return (homogenized, refs)