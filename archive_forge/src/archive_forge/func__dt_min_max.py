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
def _dt_min_max(dtype_like: npt.DTypeLike, mn: Scalar | None=None, mx: Scalar | None=None) -> tuple[Scalar, Scalar]:
    dt = np.dtype(dtype_like)
    if dt.kind in 'fc':
        dt_mn, dt_mx = (-np.inf, np.inf)
    elif dt.kind in 'iu':
        info = np.iinfo(dt)
        dt_mn, dt_mx = (info.min, info.max)
    else:
        raise ValueError('unknown dtype')
    return (dt_mn if mn is None else mn, dt_mx if mx is None else mx)