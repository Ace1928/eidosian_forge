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
def _cftimeindex_from_i8(values, date_type, name):
    """Construct a CFTimeIndex from an array of integers.

    Parameters
    ----------
    values : np.array
        Integers representing microseconds since 1970-01-01.
    date_type : cftime.datetime
        Type of date for the index.
    name : str
        Name of the index.

    Returns
    -------
    CFTimeIndex
    """
    epoch = date_type(1970, 1, 1)
    dates = np.array([epoch + timedelta(microseconds=int(value)) for value in values])
    return CFTimeIndex(dates, name=name)