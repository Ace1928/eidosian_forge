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
@pytest.mark.parametrize('calendar', _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize('units_year', [1500, 2000, 2500])
def test_use_cftime_false_non_standard_calendar(calendar, units_year) -> None:
    numerical_dates = [0, 1]
    units = f'days since {units_year}-01-01'
    with pytest.raises(OutOfBoundsDatetime):
        decode_cf_datetime(numerical_dates, units, calendar, use_cftime=False)