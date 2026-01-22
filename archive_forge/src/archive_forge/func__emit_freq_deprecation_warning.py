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
def _emit_freq_deprecation_warning(deprecated_freq):
    recommended_freq = _DEPRECATED_FREQUENICES[deprecated_freq]
    message = _DEPRECATION_MESSAGE.format(deprecated_freq=deprecated_freq, recommended_freq=recommended_freq)
    emit_user_level_warning(message, FutureWarning)