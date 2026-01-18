from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@requires_scipy
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_upsample_interpolate_bug_2197(self) -> None:
    dates = pd.date_range('2007-02-01', '2007-03-01', freq='D')
    da = xr.DataArray(np.arange(len(dates)), [('time', dates)])
    result = da.resample(time='ME').interpolate('linear')
    expected_times = np.array([np.datetime64('2007-02-28'), np.datetime64('2007-03-31')])
    expected = xr.DataArray([27.0, np.nan], [('time', expected_times)])
    assert_equal(result, expected)