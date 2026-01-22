from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
@requires_matplotlib
class TestDetermineCmapParams:

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.data = np.linspace(0, 1, num=100)

    def test_robust(self) -> None:
        cmap_params = _determine_cmap_params(self.data, robust=True)
        assert cmap_params['vmin'] == np.percentile(self.data, 2)
        assert cmap_params['vmax'] == np.percentile(self.data, 98)
        assert cmap_params['cmap'] == 'viridis'
        assert cmap_params['extend'] == 'both'
        assert cmap_params['levels'] is None
        assert cmap_params['norm'] is None

    def test_center(self) -> None:
        cmap_params = _determine_cmap_params(self.data, center=0.5)
        assert cmap_params['vmax'] - 0.5 == 0.5 - cmap_params['vmin']
        assert cmap_params['cmap'] == 'RdBu_r'
        assert cmap_params['extend'] == 'neither'
        assert cmap_params['levels'] is None
        assert cmap_params['norm'] is None

    def test_cmap_sequential_option(self) -> None:
        with xr.set_options(cmap_sequential='magma'):
            cmap_params = _determine_cmap_params(self.data)
            assert cmap_params['cmap'] == 'magma'

    def test_cmap_sequential_explicit_option(self) -> None:
        with xr.set_options(cmap_sequential=mpl.colormaps['magma']):
            cmap_params = _determine_cmap_params(self.data)
            assert cmap_params['cmap'] == mpl.colormaps['magma']

    def test_cmap_divergent_option(self) -> None:
        with xr.set_options(cmap_divergent='magma'):
            cmap_params = _determine_cmap_params(self.data, center=0.5)
            assert cmap_params['cmap'] == 'magma'

    def test_nan_inf_are_ignored(self) -> None:
        cmap_params1 = _determine_cmap_params(self.data)
        data = self.data
        data[50:55] = np.nan
        data[56:60] = np.inf
        cmap_params2 = _determine_cmap_params(data)
        assert cmap_params1['vmin'] == cmap_params2['vmin']
        assert cmap_params1['vmax'] == cmap_params2['vmax']

    @pytest.mark.slow
    def test_integer_levels(self) -> None:
        data = self.data + 1
        for level in np.arange(2, 10, dtype=int):
            cmap_params = _determine_cmap_params(data, levels=level)
            assert cmap_params['vmin'] is None
            assert cmap_params['vmax'] is None
            assert cmap_params['norm'].vmin == cmap_params['levels'][0]
            assert cmap_params['norm'].vmax == cmap_params['levels'][-1]
            assert cmap_params['extend'] == 'neither'
        cmap_params = _determine_cmap_params(data, levels=5, vmin=0, vmax=5, cmap='Blues')
        assert cmap_params['vmin'] is None
        assert cmap_params['vmax'] is None
        assert cmap_params['norm'].vmin == 0
        assert cmap_params['norm'].vmax == 5
        assert cmap_params['norm'].vmin == cmap_params['levels'][0]
        assert cmap_params['norm'].vmax == cmap_params['levels'][-1]
        assert cmap_params['cmap'].name == 'Blues'
        assert cmap_params['extend'] == 'neither'
        assert cmap_params['cmap'].N == 4
        assert cmap_params['norm'].N == 5
        cmap_params = _determine_cmap_params(data, levels=5, vmin=0.5, vmax=1.5)
        assert cmap_params['cmap'].name == 'viridis'
        assert cmap_params['extend'] == 'max'
        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.5)
        assert cmap_params['cmap'].name == 'viridis'
        assert cmap_params['extend'] == 'min'
        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.3, vmax=1.5)
        assert cmap_params['cmap'].name == 'viridis'
        assert cmap_params['extend'] == 'both'

    def test_list_levels(self) -> None:
        data = self.data + 1
        orig_levels = [0, 1, 2, 3, 4, 5]
        cmap_params = _determine_cmap_params(data, levels=orig_levels, vmin=0, vmax=3)
        assert cmap_params['vmin'] is None
        assert cmap_params['vmax'] is None
        assert cmap_params['norm'].vmin == 0
        assert cmap_params['norm'].vmax == 5
        assert cmap_params['cmap'].N == 5
        assert cmap_params['norm'].N == 6
        for wrap_levels in [list, np.array, pd.Index, DataArray]:
            cmap_params = _determine_cmap_params(data, levels=wrap_levels(orig_levels))
            assert_array_equal(cmap_params['levels'], orig_levels)

    def test_divergentcontrol(self) -> None:
        neg = self.data - 0.1
        pos = self.data
        cmap_params = _determine_cmap_params(pos)
        assert cmap_params['vmin'] == 0
        assert cmap_params['vmax'] == 1
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(neg)
        assert cmap_params['vmin'] == -0.9
        assert cmap_params['vmax'] == 0.9
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(neg, vmin=-0.1, center=False)
        assert cmap_params['vmin'] == -0.1
        assert cmap_params['vmax'] == 0.9
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(neg, vmax=0.5, center=False)
        assert cmap_params['vmin'] == -0.1
        assert cmap_params['vmax'] == 0.5
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(neg, center=False)
        assert cmap_params['vmin'] == -0.1
        assert cmap_params['vmax'] == 0.9
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(neg, center=0)
        assert cmap_params['vmin'] == -0.9
        assert cmap_params['vmax'] == 0.9
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(neg, vmin=-0.1)
        assert cmap_params['vmin'] == -0.1
        assert cmap_params['vmax'] == 0.1
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(neg, vmax=0.5)
        assert cmap_params['vmin'] == -0.5
        assert cmap_params['vmax'] == 0.5
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(neg, vmax=0.6, center=0.1)
        assert cmap_params['vmin'] == -0.4
        assert cmap_params['vmax'] == 0.6
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(pos, vmin=-0.1)
        assert cmap_params['vmin'] == -0.1
        assert cmap_params['vmax'] == 0.1
        assert cmap_params['cmap'] == 'RdBu_r'
        cmap_params = _determine_cmap_params(pos, vmin=0.1)
        assert cmap_params['vmin'] == 0.1
        assert cmap_params['vmax'] == 1
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(pos, vmax=0.5)
        assert cmap_params['vmin'] == 0
        assert cmap_params['vmax'] == 0.5
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(neg, vmin=-0.2, vmax=0.6)
        assert cmap_params['vmin'] == -0.2
        assert cmap_params['vmax'] == 0.6
        assert cmap_params['cmap'] == 'viridis'
        cmap_params = _determine_cmap_params(pos, levels=[-0.1, 0, 1])
        assert cmap_params['cmap'].name == 'RdBu_r'

    def test_norm_sets_vmin_vmax(self) -> None:
        vmin = self.data.min()
        vmax = self.data.max()
        for norm, extend, levels in zip([mpl.colors.Normalize(), mpl.colors.Normalize(), mpl.colors.Normalize(vmin + 0.1, vmax - 0.1), mpl.colors.Normalize(None, vmax - 0.1), mpl.colors.Normalize(vmin + 0.1, None)], ['neither', 'neither', 'both', 'max', 'min'], [7, None, None, None, None]):
            test_min = vmin if norm.vmin is None else norm.vmin
            test_max = vmax if norm.vmax is None else norm.vmax
            cmap_params = _determine_cmap_params(self.data, norm=norm, levels=levels)
            assert cmap_params['vmin'] is None
            assert cmap_params['vmax'] is None
            assert cmap_params['norm'].vmin == test_min
            assert cmap_params['norm'].vmax == test_max
            assert cmap_params['extend'] == extend
            assert cmap_params['norm'] == norm