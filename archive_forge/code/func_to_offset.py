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
def to_offset(freq, warn=True):
    """Convert a frequency string to the appropriate subclass of
    BaseCFTimeOffset."""
    if isinstance(freq, BaseCFTimeOffset):
        return freq
    else:
        try:
            freq_data = re.match(_PATTERN, freq).groupdict()
        except AttributeError:
            raise ValueError('Invalid frequency string provided')
    freq = freq_data['freq']
    if warn and freq in _DEPRECATED_FREQUENICES:
        _emit_freq_deprecation_warning(freq)
    multiples = freq_data['multiple']
    multiples = 1 if multiples is None else int(multiples)
    return _FREQUENCIES[freq](n=multiples)