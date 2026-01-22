from __future__ import annotations
from packaging.version import Version
import inspect
import warnings
import os
from math import isnan
import numpy as np
import pandas as pd
import xarray as xr
from datashader.utils import Expr, ngjit
from datashader.macros import expand_varargs
@staticmethod
def _expand_aggs_and_cols(append, ndims, antialiased):
    if os.environ.get('NUMBA_DISABLE_JIT', None):
        return lambda fn: fn
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            append_args = inspect.getfullargspec(append.py_func).args
        except (TypeError, AttributeError):
            append_args = inspect.getfullargspec(append).args
    append_arglen = len(append_args)
    xy_arglen = 2
    dim_arglen = ndims or 0
    aggs_and_cols_len = append_arglen - xy_arglen - dim_arglen
    if antialiased:
        aggs_and_cols_len -= 2
    return expand_varargs(aggs_and_cols_len)