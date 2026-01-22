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
class TestDiscreteColorMap:

    @pytest.fixture(autouse=True)
    def setUp(self):
        x = np.arange(start=0, stop=10, step=2)
        y = np.arange(start=9, stop=-7, step=-3)
        xy = np.dstack(np.meshgrid(x, y))
        distance = np.linalg.norm(xy, axis=2)
        self.darray = DataArray(distance, list(zip(('y', 'x'), (y, x))))
        self.data_min = distance.min()
        self.data_max = distance.max()
        yield
        plt.close('all')

    @pytest.mark.slow
    def test_recover_from_seaborn_jet_exception(self) -> None:
        pal = _color_palette('jet', 4)
        assert type(pal) == np.ndarray
        assert len(pal) == 4

    @pytest.mark.slow
    def test_build_discrete_cmap(self) -> None:
        for cmap, levels, extend, filled in [('jet', [0, 1], 'both', False), ('hot', [-4, 4], 'max', True)]:
            ncmap, cnorm = _build_discrete_cmap(cmap, levels, extend, filled)
            assert ncmap.N == len(levels) - 1
            assert len(ncmap.colors) == len(levels) - 1
            assert cnorm.N == len(levels)
            assert_array_equal(cnorm.boundaries, levels)
            assert max(levels) == cnorm.vmax
            assert min(levels) == cnorm.vmin
            if filled:
                assert ncmap.colorbar_extend == extend
            else:
                assert ncmap.colorbar_extend == 'max'

    @pytest.mark.slow
    def test_discrete_colormap_list_of_levels(self) -> None:
        for extend, levels in [('max', [-1, 2, 4, 8, 10]), ('both', [2, 5, 10, 11]), ('neither', [0, 5, 10, 15]), ('min', [2, 5, 10, 15])]:
            for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
                primitive = getattr(self.darray.plot, kind)(levels=levels)
                assert_array_equal(levels, primitive.norm.boundaries)
                assert max(levels) == primitive.norm.vmax
                assert min(levels) == primitive.norm.vmin
                if kind != 'contour':
                    assert extend == primitive.cmap.colorbar_extend
                else:
                    assert 'max' == primitive.cmap.colorbar_extend
                assert len(levels) - 1 == len(primitive.cmap.colors)

    @pytest.mark.slow
    def test_discrete_colormap_int_levels(self) -> None:
        for extend, levels, vmin, vmax, cmap in [('neither', 7, None, None, None), ('neither', 7, None, 20, mpl.colormaps['RdBu']), ('both', 7, 4, 8, None), ('min', 10, 4, 15, None)]:
            for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
                primitive = getattr(self.darray.plot, kind)(levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
                assert levels >= len(primitive.norm.boundaries) - 1
                if vmax is None:
                    assert primitive.norm.vmax >= self.data_max
                else:
                    assert primitive.norm.vmax >= vmax
                if vmin is None:
                    assert primitive.norm.vmin <= self.data_min
                else:
                    assert primitive.norm.vmin <= vmin
                if kind != 'contour':
                    assert extend == primitive.cmap.colorbar_extend
                else:
                    assert 'max' == primitive.cmap.colorbar_extend
                assert levels >= len(primitive.cmap.colors)

    def test_discrete_colormap_list_levels_and_vmin_or_vmax(self) -> None:
        levels = [0, 5, 10, 15]
        primitive = self.darray.plot(levels=levels, vmin=-3, vmax=20)
        assert primitive.norm.vmax == max(levels)
        assert primitive.norm.vmin == min(levels)

    def test_discrete_colormap_provided_boundary_norm(self) -> None:
        norm = mpl.colors.BoundaryNorm([0, 5, 10, 15], 4)
        primitive = self.darray.plot.contourf(norm=norm)
        np.testing.assert_allclose(primitive.levels, norm.boundaries)

    def test_discrete_colormap_provided_boundary_norm_matching_cmap_levels(self) -> None:
        norm = mpl.colors.BoundaryNorm([0, 5, 10, 15], 4)
        primitive = self.darray.plot.contourf(norm=norm)
        assert primitive.colorbar.norm.Ncmap == primitive.colorbar.norm.N