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
def test_resample_skipna(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    array = DataArray(np.ones(10), [('time', times)])
    array[1] = np.nan
    result = array.resample(time='1D').mean(skipna=False)
    expected = DataArray([np.nan, 1, 1], [('time', times[::4])])
    assert_identical(result, expected)