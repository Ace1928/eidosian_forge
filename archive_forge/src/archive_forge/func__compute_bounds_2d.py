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
def _compute_bounds_2d(vals):
    minval = np.inf
    maxval = -np.inf
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i][j]
            if not np.isnan(v):
                if v < minval:
                    minval = v
                if v > maxval:
                    maxval = v
    return (minval, maxval)