from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def _contains_datetime_timedeltas(array):
    """Check if an input array contains datetime.timedelta objects."""
    array = np.atleast_1d(array)
    return isinstance(array[0], timedelta)