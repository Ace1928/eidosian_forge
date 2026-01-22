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
def _generate_linear_range(start, end, periods):
    """Generate an equally-spaced sequence of cftime.datetime objects between
    and including two dates (whose length equals the number of periods)."""
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    total_seconds = (end - start).total_seconds()
    values = np.linspace(0.0, total_seconds, periods, endpoint=True)
    units = f'seconds since {format_cftime_datetime(start)}'
    calendar = start.calendar
    return cftime.num2date(values, units=units, calendar=calendar, only_use_cftime_datetimes=True)