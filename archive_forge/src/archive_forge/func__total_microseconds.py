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
def _total_microseconds(delta):
    """Compute the total number of microseconds of a datetime.timedelta.

    Parameters
    ----------
    delta : datetime.timedelta
        Input timedelta.

    Returns
    -------
    int
    """
    return delta / timedelta(microseconds=1)