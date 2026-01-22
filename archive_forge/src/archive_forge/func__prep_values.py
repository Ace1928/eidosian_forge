from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _prep_values(self, values: ArrayLike) -> np.ndarray:
    """Convert input to numpy arrays for Cython routines"""
    if needs_i8_conversion(values.dtype):
        raise NotImplementedError(f'ops for {type(self).__name__} for this dtype {values.dtype} are not implemented')
    try:
        if isinstance(values, ExtensionArray):
            values = values.to_numpy(np.float64, na_value=np.nan)
        else:
            values = ensure_float64(values)
    except (ValueError, TypeError) as err:
        raise TypeError(f'cannot handle this type -> {values.dtype}') from err
    inf = np.isinf(values)
    if inf.any():
        values = np.where(inf, np.nan, values)
    return values