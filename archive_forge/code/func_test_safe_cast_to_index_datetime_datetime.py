from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
@requires_cftime
def test_safe_cast_to_index_datetime_datetime():
    dates = [datetime(1, 1, day) for day in range(1, 20)]
    expected = pd.Index(dates)
    actual = safe_cast_to_index(np.array(dates))
    assert_array_equal(expected, actual)
    assert isinstance(actual, pd.Index)