from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import date_range_like, get_date_type
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.times import _should_cftime_be_used, convert_times
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
def _convert_to_new_calendar_with_new_day_of_year(date, day_of_year, calendar, use_cftime):
    """Convert a datetime object to another calendar with a new day of year.

    Redefines the day of year (and thus ignores the month and day information
    from the source datetime).
    Nanosecond information is lost as cftime.datetime doesn't support it.
    """
    new_date = cftime.num2date(day_of_year - 1, f'days since {date.year}-01-01', calendar=calendar if use_cftime else 'standard')
    try:
        return get_date_type(calendar, use_cftime)(date.year, new_date.month, new_date.day, date.hour, date.minute, date.second, date.microsecond)
    except ValueError:
        return np.nan