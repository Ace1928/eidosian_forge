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
def test_upsample_interpolate(self) -> None:
    from scipy.interpolate import interp1d
    xs = np.arange(6)
    ys = np.arange(3)
    times = pd.date_range('2000-01-01', freq='6h', periods=5)
    z = np.arange(5) ** 2
    data = np.tile(z, (6, 3, 1))
    array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    expected_times = times.to_series().resample('1h').asfreq().index
    new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
    kinds: list[InterpOptions] = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    for kind in kinds:
        actual = array.resample(time='1h').interpolate(kind)
        f = interp1d(np.arange(len(times)), data, kind=kind, axis=-1, bounds_error=True, assume_sorted=True)
        expected_data = f(new_times_idx)
        expected = DataArray(expected_data, {'time': expected_times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        assert_allclose(expected, actual, rtol=1e-16)