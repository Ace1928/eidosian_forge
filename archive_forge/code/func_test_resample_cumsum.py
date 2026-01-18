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
@pytest.mark.parametrize('method, expected_array', [('cumsum', [1.0, 2.0, 5.0, 6.0, 2.0, 2.0]), ('cumprod', [1.0, 2.0, 6.0, 6.0, 2.0, 2.0])])
def test_resample_cumsum(method: str, expected_array: list[float]) -> None:
    ds = xr.Dataset({'foo': ('time', [1, 2, 3, 1, 2, np.nan])}, coords={'time': xr.date_range('01-01-2001', freq='ME', periods=6, use_cftime=False)})
    actual = getattr(ds.resample(time='3ME'), method)(dim='time')
    expected = xr.Dataset({'foo': (('time',), expected_array)}, coords={'time': xr.date_range('01-01-2001', freq='ME', periods=6, use_cftime=False)})
    assert_identical(expected.drop_vars(['time']), actual)
    actual = getattr(ds.foo.resample(time='3ME'), method)(dim='time')
    expected.coords['time'] = ds.time
    assert_identical(expected.drop_vars(['time']).foo, actual)