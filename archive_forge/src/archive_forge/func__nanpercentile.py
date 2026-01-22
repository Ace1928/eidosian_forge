from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import (
def _nanpercentile(values: np.ndarray, qs: npt.NDArray[np.float64], *, na_value, mask: npt.NDArray[np.bool_], interpolation: str):
    """
    Wrapper for np.percentile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    mask : np.ndarray[bool]
        locations in values that should be considered missing
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
    if values.dtype.kind in 'mM':
        result = _nanpercentile(values.view('i8'), qs=qs, na_value=na_value.view('i8'), mask=mask, interpolation=interpolation)
        return result.astype(values.dtype)
    if mask.any():
        assert mask.shape == values.shape
        result = [_nanpercentile_1d(val, m, qs, na_value, interpolation=interpolation) for val, m in zip(list(values), list(mask))]
        if values.dtype.kind == 'f':
            result = np.array(result, dtype=values.dtype, copy=False).T
        else:
            result = np.array(result, copy=False).T
            if result.dtype != values.dtype and (not mask.all()) and (result == result.astype(values.dtype, copy=False)).all():
                result = result.astype(values.dtype, copy=False)
        return result
    else:
        return np.percentile(values, qs, axis=1, method=interpolation)