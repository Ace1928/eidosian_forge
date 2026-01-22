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
def _infer_inclusive(closed, inclusive):
    """Follows code added in pandas #43504."""
    if closed is not no_default and inclusive is not None:
        raise ValueError('Following pandas, deprecated argument `closed` cannot be passed if argument `inclusive` is not None.')
    if closed is not no_default:
        inclusive = _translate_closed_to_inclusive(closed)
    elif inclusive is None:
        inclusive = 'both'
    return inclusive