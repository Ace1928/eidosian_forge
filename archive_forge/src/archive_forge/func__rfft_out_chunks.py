from __future__ import annotations
import inspect
import warnings
from collections.abc import Sequence
import numpy as np
from dask.array.core import asarray
from dask.array.core import concatenate as _concatenate
from dask.array.creation import arange as _arange
from dask.array.numpy_compat import NUMPY_GE_200
from dask.utils import derived_from, skip_doctest
def _rfft_out_chunks(a, s, axes):
    """For computing the output chunks of rfft*"""
    if s is None:
        s = [a.chunks[axis][0] for axis in axes]
    s = list(s)
    s[-1] = s[-1] // 2 + 1
    chunks = list(a.chunks)
    for i, axis in enumerate(axes):
        chunks[axis] = (s[i],)
    return chunks