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
def test_upsample_tolerance(self) -> None:
    times = pd.date_range('2000-01-01', freq='1D', periods=2)
    times_upsampled = pd.date_range('2000-01-01', freq='6h', periods=5)
    array = DataArray(np.arange(2), [('time', times)])
    actual = array.resample(time='6h').ffill(tolerance='12h')
    expected = DataArray([0.0, 0.0, 0.0, np.nan, 1.0], [('time', times_upsampled)])
    assert_identical(expected, actual)
    actual = array.resample(time='6h').bfill(tolerance='12h')
    expected = DataArray([0.0, np.nan, 1.0, 1.0, 1.0], [('time', times_upsampled)])
    assert_identical(expected, actual)
    actual = array.resample(time='6h').nearest(tolerance='6h')
    expected = DataArray([0, 0, np.nan, 1, 1], [('time', times_upsampled)])
    assert_identical(expected, actual)