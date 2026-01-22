from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _get_take_nd_function(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt=0, mask_info=None):
    """
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    """
    func = None
    if ndim <= 2:
        func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)
    if func is None:

        def func(arr, indexer, out, fill_value=np.nan) -> None:
            indexer = ensure_platform_int(indexer)
            _take_nd_object(arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info)
    return func