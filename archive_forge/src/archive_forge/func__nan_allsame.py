import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _nan_allsame(a, axis, keepdims=False):
    """
    Determine if the values along an axis are all the same.

    nan values are ignored.

    `a` must be a numpy array.

    `axis` is assumed to be normalized; that is, 0 <= axis < a.ndim.

    For an axis of length 0, the result is True.  That is, we adopt the
    convention that ``allsame([])`` is True. (There are no values in the
    input that are different.)

    `True` is returned for slices that are all nan--not because all the
    values are the same, but because this is equivalent to ``allsame([])``.

    Examples
    --------
    >>> from numpy import nan, array
    >>> a = array([[ 3.,  3., nan,  3.],
    ...            [ 1., nan,  2.,  4.],
    ...            [nan, nan,  9., -1.],
    ...            [nan,  5.,  4.,  3.],
    ...            [ 2.,  2.,  2.,  2.],
    ...            [nan, nan, nan, nan]])
    >>> _nan_allsame(a, axis=1, keepdims=True)
    array([[ True],
           [False],
           [False],
           [False],
           [ True],
           [ True]])
    """
    if axis is None:
        if a.size == 0:
            return True
        a = a.ravel()
        axis = 0
    else:
        shp = a.shape
        if shp[axis] == 0:
            shp = shp[:axis] + (1,) * keepdims + shp[axis + 1:]
            return np.full(shp, fill_value=True, dtype=bool)
    a0 = _first_nonnan(a, axis=axis)
    return ((a0 == a) | np.isnan(a)).all(axis=axis, keepdims=keepdims)