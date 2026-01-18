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
def test_resample_and_first(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
    actual = ds.resample(time='1D').first(keep_attrs=True)
    expected = ds.isel(time=[0, 4, 8])
    assert_identical(expected, actual)
    expected_time = pd.date_range('2000-01-01', freq='3h', periods=19)
    expected = ds.reindex(time=expected_time)
    actual = ds.resample(time='3h')
    for how in ['mean', 'sum', 'first', 'last']:
        method = getattr(actual, how)
        result = method()
        assert_equal(expected, result)
    for method in [np.mean]:
        result = actual.reduce(method)
        assert_equal(expected, result)