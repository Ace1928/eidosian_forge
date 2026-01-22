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
@ngjit
def _compute_bounds_numba(arr):
    minval = np.inf
    maxval = -np.inf
    for x in arr:
        if not isnan(x):
            if x < minval:
                minval = x
            if x > maxval:
                maxval = x
    return (minval, maxval)