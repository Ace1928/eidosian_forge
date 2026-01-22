from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def _nanpolyfit_1d(arr, x, rcond=None):
    out = np.full((x.shape[1] + 1,), np.nan)
    mask = np.isnan(arr)
    if not np.all(mask):
        out[:-1], resid, rank, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
        out[-1] = resid[0] if resid.size > 0 else np.nan
        warn_on_deficient_rank(rank, x.shape[1])
    return out