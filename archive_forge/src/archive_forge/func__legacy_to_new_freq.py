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
def _legacy_to_new_freq(freq):
    if not freq or Version(pd.__version__) >= Version('2.2'):
        return freq
    try:
        freq_as_offset = to_offset(freq, warn=False)
    except ValueError:
        return freq
    if isinstance(freq_as_offset, MonthEnd) and 'ME' not in freq:
        freq = freq.replace('M', 'ME')
    elif isinstance(freq_as_offset, QuarterEnd) and 'QE' not in freq:
        freq = freq.replace('Q', 'QE')
    elif isinstance(freq_as_offset, YearBegin) and 'YS' not in freq:
        freq = freq.replace('AS', 'YS')
    elif isinstance(freq_as_offset, YearEnd):
        if 'A-' in freq:
            freq = freq.replace('A-', 'YE-')
        elif 'Y-' in freq:
            freq = freq.replace('Y-', 'YE-')
        elif freq.endswith('A'):
            freq = freq.replace('A', 'YE')
        elif 'YE' not in freq and freq.endswith('Y'):
            freq = freq.replace('Y', 'YE')
    elif isinstance(freq_as_offset, Hour):
        freq = freq.replace('H', 'h')
    elif isinstance(freq_as_offset, Minute):
        freq = freq.replace('T', 'min')
    elif isinstance(freq_as_offset, Second):
        freq = freq.replace('S', 's')
    elif isinstance(freq_as_offset, Millisecond):
        freq = freq.replace('L', 'ms')
    elif isinstance(freq_as_offset, Microsecond):
        freq = freq.replace('U', 'us')
    return freq