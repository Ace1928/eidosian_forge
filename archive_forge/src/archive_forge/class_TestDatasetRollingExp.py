from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@requires_numbagg
class TestDatasetRollingExp:

    @pytest.mark.parametrize('backend', ['numpy', pytest.param('dask', marks=requires_dask)], indirect=True)
    def test_rolling_exp(self, ds) -> None:
        result = ds.rolling_exp(time=10, window_type='span').mean()
        assert isinstance(result, Dataset)

    @pytest.mark.parametrize('backend', ['numpy'], indirect=True)
    def test_rolling_exp_keep_attrs(self, ds) -> None:
        attrs_global = {'attrs': 'global'}
        attrs_z1 = {'attr': 'z1'}
        ds.attrs = attrs_global
        ds.z1.attrs = attrs_z1
        result = ds.rolling_exp(time=10).mean()
        assert result.attrs == attrs_global
        assert result.z1.attrs == attrs_z1
        result = ds.rolling_exp(time=10).mean(keep_attrs=False)
        assert result.attrs == {}
        with set_options(keep_attrs=False):
            result = ds.rolling_exp(time=10).mean()
        assert result.attrs == {}
        with set_options(keep_attrs=False):
            result = ds.rolling_exp(time=10).mean(keep_attrs=True)
        assert result.attrs == attrs_global
        assert result.z1.attrs == attrs_z1
        with set_options(keep_attrs=True):
            result = ds.rolling_exp(time=10).mean(keep_attrs=False)
        assert result.attrs == {}
        with pytest.warns(UserWarning, match='Passing ``keep_attrs`` to ``rolling_exp`` has no effect.'):
            ds.rolling_exp(time=10, keep_attrs=True)