from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def _check_datetime_components(timestamps, timezone=None):
    from pyarrow.vendored.version import Version
    ts = pd.to_datetime(timestamps).tz_localize('UTC').tz_convert(timezone).to_series()
    tsa = pa.array(ts, pa.timestamp('ns', tz=timezone))
    subseconds = ((ts.dt.microsecond * 10 ** 3 + ts.dt.nanosecond) * 10 ** (-9)).round(9)
    iso_calendar_fields = [pa.field('iso_year', pa.int64()), pa.field('iso_week', pa.int64()), pa.field('iso_day_of_week', pa.int64())]
    if Version(pd.__version__) < Version('1.1.0'):
        iso_year = ts.map(lambda x: x.isocalendar()[0]).astype('int64')
        iso_week = ts.map(lambda x: x.isocalendar()[1]).astype('int64')
        iso_day = ts.map(lambda x: x.isocalendar()[2]).astype('int64')
    else:
        iso_year = ts.dt.isocalendar()['year'].astype('int64')
        iso_week = ts.dt.isocalendar()['week'].astype('int64')
        iso_day = ts.dt.isocalendar()['day'].astype('int64')
    iso_calendar = pa.StructArray.from_arrays([iso_year, iso_week, iso_day], fields=iso_calendar_fields)
    year = ts.dt.year.astype('int64')
    month = ts.dt.month.astype('int64')
    day = ts.dt.day.astype('int64')
    dayofweek = ts.dt.dayofweek.astype('int64')
    dayofyear = ts.dt.dayofyear.astype('int64')
    quarter = ts.dt.quarter.astype('int64')
    hour = ts.dt.hour.astype('int64')
    minute = ts.dt.minute.astype('int64')
    second = ts.dt.second.values.astype('int64')
    microsecond = ts.dt.microsecond.astype('int64')
    nanosecond = ts.dt.nanosecond.astype('int64')
    assert pc.year(tsa).equals(pa.array(year))
    assert pc.is_leap_year(tsa).equals(pa.array(ts.dt.is_leap_year))
    assert pc.month(tsa).equals(pa.array(month))
    assert pc.day(tsa).equals(pa.array(day))
    assert pc.day_of_week(tsa).equals(pa.array(dayofweek))
    assert pc.day_of_year(tsa).equals(pa.array(dayofyear))
    assert pc.iso_year(tsa).equals(pa.array(iso_year))
    assert pc.iso_week(tsa).equals(pa.array(iso_week))
    assert pc.iso_calendar(tsa).equals(iso_calendar)
    assert pc.quarter(tsa).equals(pa.array(quarter))
    assert pc.hour(tsa).equals(pa.array(hour))
    assert pc.minute(tsa).equals(pa.array(minute))
    assert pc.second(tsa).equals(pa.array(second))
    assert pc.millisecond(tsa).equals(pa.array(microsecond // 10 ** 3))
    assert pc.microsecond(tsa).equals(pa.array(microsecond % 10 ** 3))
    assert pc.nanosecond(tsa).equals(pa.array(nanosecond))
    assert pc.subsecond(tsa).equals(pa.array(subseconds))
    assert pc.local_timestamp(tsa).equals(pa.array(ts.dt.tz_localize(None)))
    if ts.dt.tz:
        if ts.dt.tz is datetime.timezone.utc:
            is_dst = [False] * len(ts)
        else:
            is_dst = ts.apply(lambda x: x.dst().seconds > 0)
        assert pc.is_dst(tsa).equals(pa.array(is_dst))
    day_of_week_options = pc.DayOfWeekOptions(count_from_zero=False, week_start=1)
    assert pc.day_of_week(tsa, options=day_of_week_options).equals(pa.array(dayofweek + 1))
    week_options = pc.WeekOptions(week_starts_monday=True, count_from_zero=False, first_week_is_fully_in_year=False)
    assert pc.week(tsa, options=week_options).equals(pa.array(iso_week))