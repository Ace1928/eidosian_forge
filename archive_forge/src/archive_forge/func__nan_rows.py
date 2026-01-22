from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _nan_rows(*arrs):
    """
    Returns a boolean array which is True where any of the rows in any
    of the _2d_ arrays in arrs are NaNs. Inputs can be any mixture of Series,
    DataFrames or array_like.
    """
    if len(arrs) == 1:
        arrs += ([[False]],)

    def _nan_row_maybe_two_inputs(x, y):
        x_is_boolean_array = hasattr(x, 'dtype') and x.dtype == bool and x
        return np.logical_or(_asarray_2d_null_rows(x), x_is_boolean_array | _asarray_2d_null_rows(y))
    return reduce(_nan_row_maybe_two_inputs, arrs).squeeze()