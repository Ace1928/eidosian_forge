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
def test_resample_by_mean_with_keep_attrs(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
    ds.attrs['dsmeta'] = 'dsdata'
    resampled_ds = ds.resample(time='1D').mean(keep_attrs=True)
    actual = resampled_ds['bar'].attrs
    expected = ds['bar'].attrs
    assert expected == actual
    actual = resampled_ds.attrs
    expected = ds.attrs
    assert expected == actual