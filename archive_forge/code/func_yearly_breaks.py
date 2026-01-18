from __future__ import annotations
import math
import typing
from datetime import datetime, timedelta, tzinfo
from typing import overload
from zoneinfo import ZoneInfo
import numpy as np
from dateutil.rrule import rrule
from ..utils import get_timezone, isclose_abs
from .date_utils import Interval, align_limits, expand_datetime_limits
from .types import DateFrequency, date_breaks_info
def yearly_breaks(info: date_breaks_info) -> Sequence[datetime]:
    """
    Calculate yearly breaks
    """
    limits = (info.start.year, info.until.year)
    l, h = align_limits(limits, info.width)
    l, h = (math.floor(l), math.ceil(h))
    _replace_d = {'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'tzinfo': info.tz}
    start = info.start.replace(year=l, **_replace_d)
    until = info.until.replace(year=h, **_replace_d)
    r = rrule(info.frequency, interval=info.width, dtstart=start, until=until)
    return list(r)