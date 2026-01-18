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
@pytest.mark.parametrize('calendar', _STANDARD_CALENDARS)
def test_use_cftime_default_standard_calendar_in_range(calendar) -> None:
    numerical_dates = [0, 1]
    units = 'days since 2000-01-01'
    expected = pd.date_range('2000', periods=2)
    with assert_no_warnings():
        result = decode_cf_datetime(numerical_dates, units, calendar)
        np.testing.assert_array_equal(result, expected)