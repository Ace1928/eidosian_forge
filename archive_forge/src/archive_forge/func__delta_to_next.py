from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
def _delta_to_next(self, last_run_at: datetime, next_hour: int, next_minute: int) -> ffwd:
    """Find next delta.

        Takes a :class:`~datetime.datetime` of last run, next minute and hour,
        and returns a :class:`~celery.utils.time.ffwd` for the next
        scheduled day and time.

        Only called when ``day_of_month`` and/or ``month_of_year``
        cronspec is specified to further limit scheduled task execution.
        """
    datedata = AttributeDict(year=last_run_at.year)
    days_of_month = sorted(self.day_of_month)
    months_of_year = sorted(self.month_of_year)

    def day_out_of_range(year: int, month: int, day: int) -> bool:
        try:
            datetime(year=year, month=month, day=day)
        except ValueError:
            return True
        return False

    def is_before_last_run(year: int, month: int, day: int) -> bool:
        return self.maybe_make_aware(datetime(year, month, day, next_hour, next_minute), naive_as_utc=False) < last_run_at

    def roll_over() -> None:
        for _ in range(2000):
            flag = datedata.dom == len(days_of_month) or day_out_of_range(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom]) or is_before_last_run(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom])
            if flag:
                datedata.dom = 0
                datedata.moy += 1
                if datedata.moy == len(months_of_year):
                    datedata.moy = 0
                    datedata.year += 1
            else:
                break
        else:
            raise RuntimeError('unable to rollover, time specification is probably invalid')
    if last_run_at.month in self.month_of_year:
        datedata.dom = bisect(days_of_month, last_run_at.day)
        datedata.moy = bisect_left(months_of_year, last_run_at.month)
    else:
        datedata.dom = 0
        datedata.moy = bisect(months_of_year, last_run_at.month)
        if datedata.moy == len(months_of_year):
            datedata.moy = 0
    roll_over()
    while 1:
        th = datetime(year=datedata.year, month=months_of_year[datedata.moy], day=days_of_month[datedata.dom])
        if th.isoweekday() % 7 in self.day_of_week:
            break
        datedata.dom += 1
        roll_over()
    return ffwd(year=datedata.year, month=months_of_year[datedata.moy], day=days_of_month[datedata.dom], hour=next_hour, minute=next_minute, second=0, microsecond=0)