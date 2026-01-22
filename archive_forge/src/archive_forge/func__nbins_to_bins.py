from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit
def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    """
    If a user passed an integer N for bins, convert this to a sequence of N
    equal(ish)-sized bins.
    """
    if is_scalar(nbins) and nbins < 1:
        raise ValueError('`bins` should be a positive integer.')
    if x_idx.size == 0:
        raise ValueError('Cannot cut empty array')
    rng = (x_idx.min(), x_idx.max())
    mn, mx = rng
    if is_numeric_dtype(x_idx.dtype) and (np.isinf(mn) or np.isinf(mx)):
        raise ValueError('cannot specify integer `bins` when input data contains infinity')
    if mn == mx:
        if _is_dt_or_td(x_idx.dtype):
            unit = dtype_to_unit(x_idx.dtype)
            td = Timedelta(seconds=1).as_unit(unit)
            bins = x_idx._values._generate_range(start=mn - td, end=mx + td, periods=nbins + 1, freq=None, unit=unit)
        else:
            mn -= 0.001 * abs(mn) if mn != 0 else 0.001
            mx += 0.001 * abs(mx) if mx != 0 else 0.001
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
    else:
        if _is_dt_or_td(x_idx.dtype):
            unit = dtype_to_unit(x_idx.dtype)
            bins = x_idx._values._generate_range(start=mn, end=mx, periods=nbins + 1, freq=None, unit=unit)
        else:
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
        adj = (mx - mn) * 0.001
        if right:
            bins[0] -= adj
        else:
            bins[-1] += adj
    return Index(bins)