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
class Common2dMixin:
    """
    Common tests for 2d plotting go here.

    These tests assume that a staticmethod for `self.plotfunc` exists.
    Should have the same name as the method.
    """
    darray: DataArray
    plotfunc: staticmethod
    pass_in_axis: Callable
    subplot_kws: dict[Any, Any] | None = None

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        da = DataArray(easy_array((10, 15), start=-1), dims=['y', 'x'], coords={'y': np.arange(10), 'x': np.arange(15)})
        ds = da.to_dataset(name='testvar')
        x, y = np.meshgrid(da.x.values, da.y.values)
        ds['x2d'] = DataArray(x, dims=['y', 'x'])
        ds['y2d'] = DataArray(y, dims=['y', 'x'])
        ds = ds.set_coords(['x2d', 'y2d'])
        self.darray: DataArray = ds.testvar
        self.darray.attrs['long_name'] = 'a_long_name'
        self.darray.attrs['units'] = 'a_units'
        self.darray.x.attrs['long_name'] = 'x_long_name'
        self.darray.x.attrs['units'] = 'x_units'
        self.darray.y.attrs['long_name'] = 'y_long_name'
        self.darray.y.attrs['units'] = 'y_units'
        self.plotmethod = getattr(self.darray.plot, self.plotfunc.__name__)

    def test_label_names(self) -> None:
        self.plotmethod()
        assert 'x_long_name [x_units]' == plt.gca().get_xlabel()
        assert 'y_long_name [y_units]' == plt.gca().get_ylabel()

    def test_1d_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match='DataArray must be 2d'):
            self.plotfunc(self.darray[0, :])

    def test_bool(self) -> None:
        xr.ones_like(self.darray, dtype=bool).plot()

    def test_complex_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match='complex128'):
            (self.darray + 1j).plot()

    def test_3d_raises_valueerror(self) -> None:
        a = DataArray(easy_array((2, 3, 4)))
        if self.plotfunc.__name__ == 'imshow':
            pytest.skip()
        with pytest.raises(ValueError, match='DataArray must be 2d'):
            self.plotfunc(a)

    def test_nonnumeric_index(self) -> None:
        a = DataArray(easy_array((3, 2)), coords=[['a', 'b', 'c'], ['d', 'e']])
        if self.plotfunc.__name__ == 'surface':
            with pytest.raises(Exception):
                self.plotfunc(a)
        else:
            self.plotfunc(a)

    def test_multiindex_raises_typeerror(self) -> None:
        a = DataArray(easy_array((3, 2)), dims=('x', 'y'), coords=dict(x=('x', [0, 1, 2]), a=('y', [0, 1]), b=('y', [2, 3])))
        a = a.set_index(y=('a', 'b'))
        with pytest.raises(TypeError, match='[Pp]lot'):
            self.plotfunc(a)

    def test_can_pass_in_axis(self) -> None:
        self.pass_in_axis(self.plotmethod)

    def test_xyincrease_defaults(self) -> None:
        self.plotfunc(DataArray(easy_array((3, 2)), coords=[[1, 2, 3], [1, 2]]))
        bounds = plt.gca().get_ylim()
        assert bounds[0] < bounds[1]
        bounds = plt.gca().get_xlim()
        assert bounds[0] < bounds[1]
        self.plotfunc(DataArray(easy_array((3, 2)), coords=[[3, 2, 1], [2, 1]]))
        bounds = plt.gca().get_ylim()
        assert bounds[0] < bounds[1]
        bounds = plt.gca().get_xlim()
        assert bounds[0] < bounds[1]

    def test_xyincrease_false_changes_axes(self) -> None:
        self.plotmethod(xincrease=False, yincrease=False)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = (xlim[0] - 14, xlim[1] - 0, ylim[0] - 9, ylim[1] - 0)
        assert all((abs(x) < 1 for x in diffs))

    def test_xyincrease_true_changes_axes(self) -> None:
        self.plotmethod(xincrease=True, yincrease=True)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = (xlim[0] - 0, xlim[1] - 14, ylim[0] - 0, ylim[1] - 9)
        assert all((abs(x) < 1 for x in diffs))

    def test_dates_are_concise(self) -> None:
        import matplotlib.dates as mdates
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.random.randn(2, len(time)), [('xx', [1, 2]), ('t', time)])
        self.plotfunc(a, x='t')
        ax = plt.gca()
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)

    def test_plot_nans(self) -> None:
        x1 = self.darray[:5]
        x2 = self.darray.copy()
        x2[5:] = np.nan
        clim1 = self.plotfunc(x1).get_clim()
        clim2 = self.plotfunc(x2).get_clim()
        assert clim1 == clim2

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore:invalid value encountered')
    def test_can_plot_all_nans(self) -> None:
        self.plotfunc(DataArray(np.full((2, 2), np.nan)))

    @pytest.mark.filterwarnings('ignore: Attempting to set')
    def test_can_plot_axis_size_one(self) -> None:
        if self.plotfunc.__name__ not in ('contour', 'contourf'):
            self.plotfunc(DataArray(np.ones((1, 1))))

    def test_disallows_rgb_arg(self) -> None:
        with pytest.raises(ValueError):
            self.plotfunc(DataArray(np.ones((2, 2))), rgb='not None')

    def test_viridis_cmap(self) -> None:
        cmap_name = self.plotmethod(cmap='viridis').get_cmap().name
        assert 'viridis' == cmap_name

    def test_default_cmap(self) -> None:
        cmap_name = self.plotmethod().get_cmap().name
        assert 'RdBu_r' == cmap_name
        cmap_name = self.plotfunc(abs(self.darray)).get_cmap().name
        assert 'viridis' == cmap_name

    @requires_seaborn
    def test_seaborn_palette_as_cmap(self) -> None:
        cmap_name = self.plotmethod(levels=2, cmap='husl').get_cmap().name
        assert 'husl' == cmap_name

    def test_can_change_default_cmap(self) -> None:
        cmap_name = self.plotmethod(cmap='Blues').get_cmap().name
        assert 'Blues' == cmap_name

    def test_diverging_color_limits(self) -> None:
        artist = self.plotmethod()
        vmin, vmax = artist.get_clim()
        assert round(abs(-vmin - vmax), 7) == 0

    def test_xy_strings(self) -> None:
        self.plotmethod(x='y', y='x')
        ax = plt.gca()
        assert 'y_long_name [y_units]' == ax.get_xlabel()
        assert 'x_long_name [x_units]' == ax.get_ylabel()

    def test_positional_coord_string(self) -> None:
        self.plotmethod(y='x')
        ax = plt.gca()
        assert 'x_long_name [x_units]' == ax.get_ylabel()
        assert 'y_long_name [y_units]' == ax.get_xlabel()
        self.plotmethod(x='x')
        ax = plt.gca()
        assert 'x_long_name [x_units]' == ax.get_xlabel()
        assert 'y_long_name [y_units]' == ax.get_ylabel()

    def test_bad_x_string_exception(self) -> None:
        with pytest.raises(ValueError, match='x and y cannot be equal.'):
            self.plotmethod(x='y', y='y')
        error_msg = "must be one of None, 'x', 'x2d', 'y', 'y2d'"
        with pytest.raises(ValueError, match=f'x {error_msg}'):
            self.plotmethod(x='not_a_real_dim', y='y')
        with pytest.raises(ValueError, match=f'x {error_msg}'):
            self.plotmethod(x='not_a_real_dim')
        with pytest.raises(ValueError, match=f'y {error_msg}'):
            self.plotmethod(y='not_a_real_dim')
        self.darray.coords['z'] = 100

    def test_coord_strings(self) -> None:
        assert {'x', 'y'} == set(self.darray.dims)
        self.plotmethod(y='y', x='x')

    def test_non_linked_coords(self) -> None:
        self.darray.coords['newy'] = self.darray.y + 150
        self.plotfunc(self.darray, x='x', y='newy')
        ax = plt.gca()
        assert 'x_long_name [x_units]' == ax.get_xlabel()
        assert 'newy' == ax.get_ylabel()
        assert np.min(ax.get_ylim()) > 100.0

    def test_non_linked_coords_transpose(self) -> None:
        self.darray.coords['newy'] = self.darray.y + 150
        self.plotfunc(self.darray, x='newy', y='x')
        ax = plt.gca()
        assert 'newy' == ax.get_xlabel()
        assert 'x_long_name [x_units]' == ax.get_ylabel()
        assert np.min(ax.get_xlim()) > 100.0

    def test_multiindex_level_as_coord(self) -> None:
        da = DataArray(easy_array((3, 2)), dims=('x', 'y'), coords=dict(x=('x', [0, 1, 2]), a=('y', [0, 1]), b=('y', [2, 3])))
        da = da.set_index(y=['a', 'b'])
        for x, y in (('a', 'x'), ('b', 'x'), ('x', 'a'), ('x', 'b')):
            self.plotfunc(da, x=x, y=y)
            ax = plt.gca()
            assert x == ax.get_xlabel()
            assert y == ax.get_ylabel()
        with pytest.raises(ValueError, match='levels of the same MultiIndex'):
            self.plotfunc(da, x='a', y='b')
        with pytest.raises(ValueError, match="y must be one of None, 'a', 'b', 'x'"):
            self.plotfunc(da, x='a', y='y')

    def test_default_title(self) -> None:
        a = DataArray(easy_array((4, 3, 2)), dims=['a', 'b', 'c'])
        a.coords['c'] = [0, 1]
        a.coords['d'] = 'foo'
        self.plotfunc(a.isel(c=1))
        title = plt.gca().get_title()
        assert 'c = 1, d = foo' == title or 'd = foo, c = 1' == title

    def test_colorbar_default_label(self) -> None:
        self.plotmethod(add_colorbar=True)
        assert 'a_long_name [a_units]' in text_in_fig()

    def test_no_labels(self) -> None:
        self.darray.name = 'testvar'
        self.darray.attrs['units'] = 'test_units'
        self.plotmethod(add_labels=False)
        alltxt = text_in_fig()
        for string in ['x_long_name [x_units]', 'y_long_name [y_units]', 'testvar [test_units]']:
            assert string not in alltxt

    def test_colorbar_kwargs(self) -> None:
        self.darray.attrs.pop('long_name')
        self.darray.attrs['units'] = 'test_units'
        self.plotmethod(add_colorbar=True)
        alltxt = text_in_fig()
        assert 'testvar [test_units]' in alltxt
        self.darray.attrs.pop('units')
        self.darray.name = 'testvar'
        self.plotmethod(add_colorbar=True, cbar_kwargs={'label': 'MyLabel'})
        alltxt = text_in_fig()
        assert 'MyLabel' in alltxt
        assert 'testvar' not in alltxt
        self.plotmethod(add_colorbar=True, cbar_kwargs=(('label', 'MyLabel'),))
        alltxt = text_in_fig()
        assert 'MyLabel' in alltxt
        assert 'testvar' not in alltxt
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(ax=ax, cbar_ax=cax, add_colorbar=True, cbar_kwargs={'label': 'MyBar'})
        assert ax.has_data()
        assert cax.has_data()
        alltxt = text_in_fig()
        assert 'MyBar' in alltxt
        assert 'testvar' not in alltxt
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(ax=ax, add_colorbar=True, cbar_kwargs={'label': 'MyBar', 'cax': cax})
        assert ax.has_data()
        assert cax.has_data()
        alltxt = text_in_fig()
        assert 'MyBar' in alltxt
        assert 'testvar' not in alltxt
        self.plotmethod(add_colorbar=False)
        assert 'testvar' not in text_in_fig()
        pytest.raises(ValueError, self.plotmethod, add_colorbar=False, cbar_kwargs={'label': 'label'})

    def test_verbose_facetgrid(self) -> None:
        a = easy_array((10, 15, 3))
        d = DataArray(a, dims=['y', 'x', 'z'])
        g = xplt.FacetGrid(d, col='z', subplot_kws=self.subplot_kws)
        g.map_dataarray(self.plotfunc, 'x', 'y')
        for ax in g.axs.flat:
            assert ax.has_data()

    def test_2d_function_and_method_signature_same(self) -> None:
        func_sig = inspect.signature(self.plotfunc)
        method_sig = inspect.signature(self.plotmethod)
        for argname, param in method_sig.parameters.items():
            assert func_sig.parameters[argname] == param

    @pytest.mark.filterwarnings('ignore:tight_layout cannot')
    def test_convenient_facetgrid(self) -> None:
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        g = self.plotfunc(d, x='x', y='y', col='z', col_wrap=2)
        assert_array_equal(g.axs.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axs):
            assert ax.has_data()
            if x == 0:
                assert 'y' == ax.get_ylabel()
            else:
                assert '' == ax.get_ylabel()
            if y == 1:
                assert 'x' == ax.get_xlabel()
            else:
                assert '' == ax.get_xlabel()
        g = self.plotfunc(d, col='z', col_wrap=2)
        assert_array_equal(g.axs.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axs):
            assert ax.has_data()
            if x == 0:
                assert 'y' == ax.get_ylabel()
            else:
                assert '' == ax.get_ylabel()
            if y == 1:
                assert 'x' == ax.get_xlabel()
            else:
                assert '' == ax.get_xlabel()

    @pytest.mark.filterwarnings('ignore:tight_layout cannot')
    def test_convenient_facetgrid_4d(self) -> None:
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = self.plotfunc(d, x='x', y='y', col='columns', row='rows')
        assert_array_equal(g.axs.shape, [3, 2])
        for ax in g.axs.flat:
            assert ax.has_data()

    @pytest.mark.filterwarnings('ignore:This figure includes')
    def test_facetgrid_map_only_appends_mappables(self) -> None:
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = self.plotfunc(d, x='x', y='y', col='columns', row='rows')
        expected = g._mappables
        g.map(lambda: plt.plot(1, 1))
        actual = g._mappables
        assert expected == actual

    def test_facetgrid_cmap(self) -> None:
        data = np.random.random(size=(20, 25, 12)) + np.linspace(-3, 3, 12)
        d = DataArray(data, dims=['x', 'y', 'time'])
        fg = d.plot.pcolormesh(col='time')
        assert len({m.get_clim() for m in fg._mappables}) == 1
        assert len({m.get_cmap().name for m in fg._mappables}) == 1

    def test_facetgrid_cbar_kwargs(self) -> None:
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = self.plotfunc(d, x='x', y='y', col='columns', row='rows', cbar_kwargs={'label': 'test_label'})
        if g.cbar is not None:
            assert get_colorbar_label(g.cbar) == 'test_label'

    def test_facetgrid_no_cbar_ax(self) -> None:
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        with pytest.raises(ValueError):
            self.plotfunc(d, x='x', y='y', col='columns', row='rows', cbar_ax=1)

    def test_cmap_and_color_both(self) -> None:
        with pytest.raises(ValueError):
            self.plotmethod(colors='k', cmap='RdBu')

    def test_2d_coord_with_interval(self) -> None:
        for dim in self.darray.dims:
            gp = self.darray.groupby_bins(dim, range(15), restore_coord_dims=True).mean([dim])
            for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
                getattr(gp.plot, kind)()

    def test_colormap_error_norm_and_vmin_vmax(self) -> None:
        norm = mpl.colors.LogNorm(0.1, 10.0)
        with pytest.raises(ValueError):
            self.darray.plot(norm=norm, vmin=2)
        with pytest.raises(ValueError):
            self.darray.plot(norm=norm, vmax=2)