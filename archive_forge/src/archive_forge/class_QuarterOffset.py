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
class QuarterOffset(BaseCFTimeOffset):
    """Quarter representation copied off of pandas/tseries/offsets.py"""
    _freq: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        months_since = other.month % 3 - self.month % 3
        qtrs = roll_qtrday(other, self.n, self.month, day_option=self._day_option, modby=3)
        months = qtrs * 3 - months_since
        return _shift_month(other, months, self._day_option)

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        mod_month = (date.month - self.month) % 3
        return mod_month == 0 and date.day == self._get_offset_day(date)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        return f'{self._freq}-{_MONTH_ABBREVIATIONS[self.month]}'

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'