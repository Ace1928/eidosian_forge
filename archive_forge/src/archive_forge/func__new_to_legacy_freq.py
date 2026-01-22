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
def _new_to_legacy_freq(freq):
    if not freq or Version(pd.__version__) >= Version('2.2'):
        return freq
    try:
        freq_as_offset = to_offset(freq)
    except ValueError:
        return freq
    if isinstance(freq_as_offset, MonthEnd) and 'ME' in freq:
        freq = freq.replace('ME', 'M')
    elif isinstance(freq_as_offset, QuarterEnd) and 'QE' in freq:
        freq = freq.replace('QE', 'Q')
    elif isinstance(freq_as_offset, YearBegin) and 'YS' in freq:
        freq = freq.replace('YS', 'AS')
    elif isinstance(freq_as_offset, YearEnd):
        if 'Y-' in freq:
            freq = freq.replace('Y-', 'A-')
        elif 'YE-' in freq:
            freq = freq.replace('YE-', 'A-')
        elif 'A-' not in freq and freq.endswith('Y'):
            freq = freq.replace('Y', 'A')
        elif freq.endswith('YE'):
            freq = freq.replace('YE', 'A')
    return freq