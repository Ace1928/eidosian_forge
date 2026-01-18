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
def test_resample_drop_nondim_coords(self) -> None:
    xs = np.arange(6)
    ys = np.arange(3)
    times = pd.date_range('2000-01-01', freq='6h', periods=5)
    data = np.tile(np.arange(5), (6, 3, 1))
    xx, yy = np.meshgrid(xs * 5, ys * 2.5)
    tt = np.arange(len(times), dtype=int)
    array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
    xcoord = DataArray(xx.T, {'x': xs, 'y': ys}, ('x', 'y'))
    ycoord = DataArray(yy.T, {'x': xs, 'y': ys}, ('x', 'y'))
    tcoord = DataArray(tt, {'time': times}, ('time',))
    ds = Dataset({'data': array, 'xc': xcoord, 'yc': ycoord, 'tc': tcoord})
    ds = ds.set_coords(['xc', 'yc', 'tc'])
    actual = ds.resample(time='12h').mean('time')
    assert 'tc' not in actual.coords
    actual = ds.resample(time='1h').ffill()
    assert 'tc' not in actual.coords
    actual = ds.resample(time='1h').interpolate('linear')
    assert 'tc' not in actual.coords