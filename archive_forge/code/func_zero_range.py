from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def zero_range(x: tuple[Any, Any], tol: float=EPSILON * 100) -> bool:
    """
    Determine if range of vector is close to zero.

    Parameters
    ----------
    x : array_like
        Value(s) to check. If it is an array_like, it
        should be of length 2.
    tol : float
        Tolerance. Default tolerance is the `machine epsilon`_
        times :math:`10^2`.

    Returns
    -------
    out : bool
        Whether ``x`` has zero range.

    Examples
    --------
    >>> zero_range([1, 1])
    True
    >>> zero_range([1, 2])
    False
    >>> zero_range([1, 2], tol=2)
    True

    .. _machine epsilon: https://en.wikipedia.org/wiki/Machine_epsilon
    """
    if x[0] > x[1]:
        x = (x[1], x[0])
    if isinstance(x[0], (pd.Timestamp, datetime.datetime)):
        from mizani._core.dates import datetime_to_num
        l, h = datetime_to_num(x)
        return l == h
    elif isinstance(x[0], np.datetime64):
        return x[0] == x[1]
    elif isinstance(x[0], (pd.Timedelta, datetime.timedelta)):
        return x[0].total_seconds() == x[1].total_seconds()
    elif isinstance(x[0], np.timedelta64):
        return x[0] == x[1]
    elif not isinstance(x[0], (float, int, np.number)):
        raise TypeError("zero_range objects cannot work with objects of type '{}'".format(type(x[0])))
    else:
        low, high = x
    if any(np.isnan((low, high))):
        return True
    if low == high:
        return True
    if any(np.isinf((low, high))):
        return False
    low_abs = np.abs(low)
    if low_abs == 0:
        return False
    return (high - low) / low_abs < tol