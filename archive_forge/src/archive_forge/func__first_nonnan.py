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
def _first_nonnan(a, axis):
    """
    Return the first non-nan value along the given axis.

    If a slice is all nan, nan is returned for that slice.

    The shape of the return value corresponds to ``keepdims=True``.

    Examples
    --------
    >>> import numpy as np
    >>> nan = np.nan
    >>> a = np.array([[ 3.,  3., nan,  3.],
                      [ 1., nan,  2.,  4.],
                      [nan, nan,  9., -1.],
                      [nan,  5.,  4.,  3.],
                      [ 2.,  2.,  2.,  2.],
                      [nan, nan, nan, nan]])
    >>> _first_nonnan(a, axis=0)
    array([[3., 3., 2., 3.]])
    >>> _first_nonnan(a, axis=1)
    array([[ 3.],
           [ 1.],
           [ 9.],
           [ 5.],
           [ 2.],
           [nan]])
    """
    k = _argmin(np.isnan(a), axis=axis, keepdims=True)
    return np.take_along_axis(a, k, axis=axis)