from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count

    Reduction for 1D masked array.

    Parameters
    ----------
    func : np.min or np.max
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    axis : int, optional, default None
    