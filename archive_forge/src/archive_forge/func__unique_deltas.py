from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def _unique_deltas(arr):
    """Sorted unique deltas of numpy array"""
    return np.sort(np.unique(np.diff(arr)))