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
class TestPlotStep(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_step(self) -> None:
        hdl = self.darray[0, 0].plot.step()
        assert 'steps' in hdl[0].get_drawstyle()

    @pytest.mark.parametrize('where', ['pre', 'post', 'mid'])
    def test_step_with_where(self, where) -> None:
        hdl = self.darray[0, 0].plot.step(where=where)
        assert hdl[0].get_drawstyle() == f'steps-{where}'

    def test_step_with_hue(self) -> None:
        hdl = self.darray[0].plot.step(hue='dim_2')
        assert hdl[0].get_drawstyle() == 'steps-pre'

    @pytest.mark.parametrize('where', ['pre', 'post', 'mid'])
    def test_step_with_hue_and_where(self, where) -> None:
        hdl = self.darray[0].plot.step(hue='dim_2', where=where)
        assert hdl[0].get_drawstyle() == f'steps-{where}'

    def test_drawstyle_steps(self) -> None:
        hdl = self.darray[0].plot(hue='dim_2', drawstyle='steps')
        assert hdl[0].get_drawstyle() == 'steps'

    @pytest.mark.parametrize('where', ['pre', 'post', 'mid'])
    def test_drawstyle_steps_with_where(self, where) -> None:
        hdl = self.darray[0].plot(hue='dim_2', drawstyle=f'steps-{where}')
        assert hdl[0].get_drawstyle() == f'steps-{where}'

    def test_coord_with_interval_step(self) -> None:
        """Test step plot with intervals."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot.step()
        line = plt.gca().lines[0]
        assert isinstance(line, mpl.lines.Line2D)
        assert len(np.asarray(line.get_xdata())) == (len(bins) - 1) * 2

    def test_coord_with_interval_step_x(self) -> None:
        """Test step plot with intervals explicitly on x axis."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot.step(x='dim_0_bins')
        line = plt.gca().lines[0]
        assert isinstance(line, mpl.lines.Line2D)
        assert len(np.asarray(line.get_xdata())) == (len(bins) - 1) * 2

    def test_coord_with_interval_step_y(self) -> None:
        """Test step plot with intervals explicitly on y axis."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot.step(y='dim_0_bins')
        line = plt.gca().lines[0]
        assert isinstance(line, mpl.lines.Line2D)
        assert len(np.asarray(line.get_xdata())) == (len(bins) - 1) * 2

    def test_coord_with_interval_step_x_and_y_raises_valueeerror(self) -> None:
        """Test that step plot with intervals both on x and y axes raises an error."""
        arr = xr.DataArray([pd.Interval(0, 1), pd.Interval(1, 2)], coords=[('x', [pd.Interval(0, 1), pd.Interval(1, 2)])])
        with pytest.raises(TypeError, match='intervals against intervals'):
            arr.plot.step()