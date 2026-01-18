from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('funcname, argument', [('reduce', (np.mean,)), ('mean', ()), ('construct', ('window_dim',)), ('count', ())])
def test_rolling_keep_attrs(self, funcname, argument) -> None:
    global_attrs = {'units': 'test', 'long_name': 'testing'}
    da_attrs = {'da_attr': 'test'}
    da_not_rolled_attrs = {'da_not_rolled_attr': 'test'}
    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)
    ds = Dataset(data_vars={'da': ('coord', data), 'da_not_rolled': ('no_coord', data)}, coords={'coord': coords}, attrs=global_attrs)
    ds.da.attrs = da_attrs
    ds.da_not_rolled.attrs = da_not_rolled_attrs
    func = getattr(ds.rolling(dim={'coord': 5}), funcname)
    result = func(*argument)
    assert result.attrs == global_attrs
    assert result.da.attrs == da_attrs
    assert result.da_not_rolled.attrs == da_not_rolled_attrs
    assert result.da.name == 'da'
    assert result.da_not_rolled.name == 'da_not_rolled'
    func = getattr(ds.rolling(dim={'coord': 5}), funcname)
    result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == 'da'
    assert result.da_not_rolled.name == 'da_not_rolled'
    func = getattr(ds.rolling(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument)
    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == 'da'
    assert result.da_not_rolled.name == 'da_not_rolled'
    func = getattr(ds.rolling(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument, keep_attrs=True)
    assert result.attrs == global_attrs
    assert result.da.attrs == da_attrs
    assert result.da_not_rolled.attrs == da_not_rolled_attrs
    assert result.da.name == 'da'
    assert result.da_not_rolled.name == 'da_not_rolled'
    func = getattr(ds.rolling(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=True):
        result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == 'da'
    assert result.da_not_rolled.name == 'da_not_rolled'