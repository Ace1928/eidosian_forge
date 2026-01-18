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
def test_upsample_nd(self) -> None:
    xs = np.arange(6)
    ys = np.arange(3)
    times = pd.date_range('2000-01-01', freq='6h', periods=5)
    data = np.tile(np.arange(5), (6, 3, 1))
    array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    actual = array.resample(time='3h').ffill()
    expected_data = np.repeat(data, 2, axis=-1)
    expected_times = times.to_series().resample('3h').asfreq().index
    expected_data = expected_data[..., :len(expected_times)]
    expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    assert_identical(expected, actual)
    actual = array.resample(time='3h').ffill()
    expected_data = np.repeat(np.flipud(data.T).T, 2, axis=-1)
    expected_data = np.flipud(expected_data.T).T
    expected_times = times.to_series().resample('3h').asfreq().index
    expected_data = expected_data[..., :len(expected_times)]
    expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    assert_identical(expected, actual)
    actual = array.resample(time='3h').asfreq()
    expected_data = np.repeat(data, 2, axis=-1).astype(float)[..., :-1]
    expected_data[..., 1::2] = np.nan
    expected_times = times.to_series().resample('3h').asfreq().index
    expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    assert_identical(expected, actual)
    actual = array.resample(time='3h').pad()
    expected_data = np.repeat(data, 2, axis=-1)
    expected_data[..., 1::2] = expected_data[..., ::2]
    expected_data = expected_data[..., :-1]
    expected_times = times.to_series().resample('3h').asfreq().index
    expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    assert_identical(expected, actual)