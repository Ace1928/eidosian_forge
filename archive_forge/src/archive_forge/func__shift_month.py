from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import (
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import (
from xarray.core.utils import emit_user_level_warning
def _shift_month(date, months, day_option='start'):
    """Shift the date to a month start or end a given number of months away."""
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    delta_year = (date.month + months) // 12
    month = (date.month + months) % 12
    if month == 0:
        month = 12
        delta_year = delta_year - 1
    year = date.year + delta_year
    if day_option == 'start':
        day = 1
    elif day_option == 'end':
        reference = type(date)(year, month, 1)
        day = _days_in_month(reference)
    else:
        raise ValueError(day_option)
    return date.replace(year=year, month=month, day=day)