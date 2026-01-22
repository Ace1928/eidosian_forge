from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def _round_to_nearest_half_even(values, unit):
    """Copied from pandas."""
    if unit % 2:
        return _ceil_int(values - unit // 2, unit)
    quotient, remainder = np.divmod(values, unit)
    mask = np.logical_or(remainder > unit // 2, np.logical_and(remainder == unit // 2, quotient % 2))
    quotient[mask] += 1
    return quotient * unit