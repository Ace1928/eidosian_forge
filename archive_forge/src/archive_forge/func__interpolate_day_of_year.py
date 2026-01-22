from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import date_range_like, get_date_type
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.times import _should_cftime_be_used, convert_times
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
def _interpolate_day_of_year(time, target_calendar, use_cftime):
    """Returns the nearest day in the target calendar of the corresponding
    "decimal year" in the source calendar.
    """
    year = int(time.dt.year[0])
    source_calendar = time.dt.calendar
    return np.round(_days_in_year(year, target_calendar, use_cftime) * time.dt.dayofyear / _days_in_year(year, source_calendar, use_cftime)).astype(int)