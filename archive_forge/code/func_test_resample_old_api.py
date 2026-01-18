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
def test_resample_old_api(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
    with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
        ds.resample('1D', 'time')
    with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
        ds.resample('1D', dim='time', how='mean')
    with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
        ds.resample('1D', dim='time')