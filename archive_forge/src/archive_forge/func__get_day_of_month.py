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
def _get_day_of_month(other, day_option):
    """Find the day in `other`'s month that satisfies a BaseCFTimeOffset's
    onOffset policy, as described by the `day_option` argument.

    Parameters
    ----------
    other : cftime.datetime
    day_option : 'start', 'end'
        'start': returns 1
        'end': returns last day of the month

    Returns
    -------
    day_of_month : int

    """
    if day_option == 'start':
        return 1
    elif day_option == 'end':
        return _days_in_month(other)
    elif day_option is None:
        raise NotImplementedError()
    else:
        raise ValueError(day_option)