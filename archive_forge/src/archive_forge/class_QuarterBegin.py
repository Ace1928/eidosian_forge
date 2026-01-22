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
class QuarterBegin(QuarterOffset):
    _default_month = 3
    _freq = 'QS'
    _day_option = 'start'

    def rollforward(self, date):
        """Roll date forward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date + QuarterBegin(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date - QuarterBegin(month=self.month)