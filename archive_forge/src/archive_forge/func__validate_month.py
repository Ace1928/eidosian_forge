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
def _validate_month(month, default_month):
    result_month = default_month if month is None else month
    if not isinstance(result_month, int):
        raise TypeError(f"'self.month' must be an integer value between 1 and 12.  Instead, it was set to a value of {result_month!r}")
    elif not 1 <= result_month <= 12:
        raise ValueError(f"'self.month' must be an integer value between 1 and 12.  Instead, it was set to a value of {result_month!r}")
    return result_month