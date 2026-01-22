from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def _matching_float(np_type: npt.DTypeLike) -> type[np.floating]:
    """Return floating point type matching `np_type`"""
    dtype = np.dtype(np_type)
    if dtype.kind not in 'cf':
        raise ValueError('Expecting float or complex type as input')
    if issubclass(dtype.type, np.floating):
        return dtype.type
    return _CSIZE2FLOAT[dtype.itemsize]