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
def _round_via_method(self, freq, method):
    """Round dates using a specified method."""
    from xarray.coding.cftime_offsets import CFTIME_TICKS, to_offset
    if not self._data.size:
        return CFTimeIndex(np.array(self))
    offset = to_offset(freq)
    if not isinstance(offset, CFTIME_TICKS):
        raise ValueError(f'{offset} is a non-fixed frequency')
    unit = _total_microseconds(offset.as_timedelta())
    values = self.asi8
    rounded = method(values, unit)
    return _cftimeindex_from_i8(rounded, self.date_type, self.name)