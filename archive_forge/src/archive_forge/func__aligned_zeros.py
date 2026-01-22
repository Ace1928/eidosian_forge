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
def _aligned_zeros(shape, dtype=float, order='C', align=None):
    """Allocate a new ndarray with aligned memory.

    Primary use case for this currently is working around a f2py issue
    in NumPy 1.9.1, where dtype.alignment is such that np.zeros() does
    not necessarily create arrays aligned up to it.

    """
    dtype = np.dtype(dtype)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + align + 1, np.uint8)
    offset = buf.__array_interface__['data'][0] % align
    if offset != 0:
        offset = align - offset
    buf = buf[offset:offset + size + 1][:-1]
    data = np.ndarray(shape, dtype, buf, order=order)
    data.fill(0)
    return data