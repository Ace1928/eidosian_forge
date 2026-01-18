from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
def test_strftime_of_cftime_array(calendar):
    date_format = '%Y%m%d%H%M'
    cf_values = xr.cftime_range('2000', periods=5, calendar=calendar)
    dt_values = pd.date_range('2000', periods=5)
    expected = pd.Index(dt_values.strftime(date_format))
    result = cf_values.strftime(date_format)
    assert result.equals(expected)