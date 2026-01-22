from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _nan_row_maybe_two_inputs(x, y):
    x_is_boolean_array = hasattr(x, 'dtype') and x.dtype == bool and x
    return np.logical_or(_asarray_2d_null_rows(x), x_is_boolean_array | _asarray_2d_null_rows(y))