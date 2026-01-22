import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
class RRuleLocator(DateLocator):

    def __init__(self, o, tz=None):
        super().__init__(tz)
        self.rule = o

    def __call__(self):
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []
        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        start, stop = self._create_rrule(vmin, vmax)
        dates = self.rule.between(start, stop, True)
        if len(dates) == 0:
            return date2num([vmin, vmax])
        return self.raise_if_exceeds(date2num(dates))

    def _create_rrule(self, vmin, vmax):
        delta = relativedelta(vmax, vmin)
        try:
            start = vmin - delta
        except (ValueError, OverflowError):
            start = datetime.datetime(1, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        try:
            stop = vmax + delta
        except (ValueError, OverflowError):
            stop = datetime.datetime(9999, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
        self.rule.set(dtstart=start, until=stop)
        return (vmin, vmax)

    def _get_unit(self):
        freq = self.rule._rrule._freq
        return self.get_unit_generic(freq)

    @staticmethod
    def get_unit_generic(freq):
        if freq == YEARLY:
            return DAYS_PER_YEAR
        elif freq == MONTHLY:
            return DAYS_PER_MONTH
        elif freq == WEEKLY:
            return DAYS_PER_WEEK
        elif freq == DAILY:
            return 1.0
        elif freq == HOURLY:
            return 1.0 / HOURS_PER_DAY
        elif freq == MINUTELY:
            return 1.0 / MINUTES_PER_DAY
        elif freq == SECONDLY:
            return 1.0 / SEC_PER_DAY
        else:
            return -1

    def _get_interval(self):
        return self.rule._rrule._interval