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
def _generate_anchored_offsets(base_freq, offset):
    offsets = {}
    for month, abbreviation in _MONTH_ABBREVIATIONS.items():
        anchored_freq = f'{base_freq}-{abbreviation}'
        offsets[anchored_freq] = partial(offset, month=month)
    return offsets