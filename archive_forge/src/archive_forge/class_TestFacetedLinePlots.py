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
@pytest.mark.filterwarnings('ignore:tight_layout cannot')
class TestFacetedLinePlots(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.darray = DataArray(np.random.randn(10, 6, 3, 4), dims=['hue', 'x', 'col', 'row'], coords=[range(10), range(6), range(3), ['A', 'B', 'C', 'C++']], name='Cornelius Ortega the 1st')
        self.darray.hue.name = 'huename'
        self.darray.hue.attrs['units'] = 'hunits'
        self.darray.x.attrs['units'] = 'xunits'
        self.darray.col.attrs['units'] = 'colunits'
        self.darray.row.attrs['units'] = 'rowunits'

    def test_facetgrid_shape(self) -> None:
        g = self.darray.plot(row='row', col='col', hue='hue')
        assert g.axs.shape == (len(self.darray.row), len(self.darray.col))
        g = self.darray.plot(row='col', col='row', hue='hue')
        assert g.axs.shape == (len(self.darray.col), len(self.darray.row))

    def test_unnamed_args(self) -> None:
        g = self.darray.plot.line('o--', row='row', col='col', hue='hue')
        lines = [q for q in g.axs.flat[0].get_children() if isinstance(q, mpl.lines.Line2D)]
        assert lines[0].get_marker() == 'o'
        assert lines[0].get_linestyle() == '--'

    def test_default_labels(self) -> None:
        g = self.darray.plot(row='row', col='col', hue='hue')
        for label, ax in zip(self.darray.coords['row'].values, g.axs[:, -1]):
            assert substring_in_axes(label, ax)
        for label, ax in zip(self.darray.coords['col'].values, g.axs[0, :]):
            assert substring_in_axes(str(label), ax)
        for ax in g.axs[:, 0]:
            assert substring_in_axes(self.darray.name, ax)

    def test_test_empty_cell(self) -> None:
        g = self.darray.isel(row=1).drop_vars('row').plot(col='col', hue='hue', col_wrap=2)
        bottomright = g.axs[-1, -1]
        assert not bottomright.has_data()
        assert not bottomright.get_visible()

    def test_set_axis_labels(self) -> None:
        g = self.darray.plot(row='row', col='col', hue='hue')
        g.set_axis_labels('longitude', 'latitude')
        alltxt = text_in_fig()
        assert 'longitude' in alltxt
        assert 'latitude' in alltxt

    def test_axes_in_faceted_plot(self) -> None:
        with pytest.raises(ValueError):
            self.darray.plot.line(row='row', col='col', x='x', ax=plt.axes())

    def test_figsize_and_size(self) -> None:
        with pytest.raises(ValueError):
            self.darray.plot.line(row='row', col='col', x='x', size=3, figsize=(4, 3))

    def test_wrong_num_of_dimensions(self) -> None:
        with pytest.raises(ValueError):
            self.darray.plot(row='row', hue='hue')
            self.darray.plot.line(row='row', hue='hue')