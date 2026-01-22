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
def _parsed_string_to_bounds(date_type, resolution, parsed):
    """Generalization of
    pandas.tseries.index.DatetimeIndex._parsed_string_to_bounds
    for use with non-standard calendars and cftime.datetime
    objects.
    """
    if resolution == 'year':
        return (date_type(parsed.year, 1, 1), date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1))
    elif resolution == 'month':
        if parsed.month == 12:
            end = date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = date_type(parsed.year, parsed.month + 1, 1) - timedelta(microseconds=1)
        return (date_type(parsed.year, parsed.month, 1), end)
    elif resolution == 'day':
        start = date_type(parsed.year, parsed.month, parsed.day)
        return (start, start + timedelta(days=1, microseconds=-1))
    elif resolution == 'hour':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour)
        return (start, start + timedelta(hours=1, microseconds=-1))
    elif resolution == 'minute':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute)
        return (start, start + timedelta(minutes=1, microseconds=-1))
    elif resolution == 'second':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
        return (start, start + timedelta(seconds=1, microseconds=-1))
    else:
        raise KeyError