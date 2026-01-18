from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@requires_cftime
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
@pytest.mark.parametrize('units_year', [1500, 2000, 2500])
def test_use_cftime_true(calendar, units_year) -> None:
    from cftime import num2date
    numerical_dates = [0, 1]
    units = f'days since {units_year}-01-01'
    expected = num2date(numerical_dates, units, calendar, only_use_cftime_datetimes=True)
    with assert_no_warnings():
        result = decode_cf_datetime(numerical_dates, units, calendar, use_cftime=True)
        np.testing.assert_array_equal(result, expected)