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
class YearEnd(YearOffset):
    _freq = 'YE'
    _day_option = 'end'
    _default_month = 12

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date) and date.month == self.month

    def rollforward(self, date):
        """Roll date forward to nearest end of year"""
        if self.onOffset(date):
            return date
        else:
            return date + YearEnd(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest end of year"""
        if self.onOffset(date):
            return date
        else:
            return date - YearEnd(month=self.month)