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
def test_resample_offset(self) -> None:
    times = pd.date_range('2000-01-01T02:03:01', freq='6h', periods=10)
    array = DataArray(np.arange(10), [('time', times)])
    offset = pd.Timedelta('11h')
    actual = array.resample(time='24h', offset=offset).mean()
    expected = DataArray(array.to_series().resample('24h', offset=offset).mean())
    assert_identical(expected, actual)