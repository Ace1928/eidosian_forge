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
class TestDatetimePlot(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """
        Create a DataArray with a time-axis that contains datetime objects.
        """
        month = np.arange(1, 13, 1)
        data = np.sin(2 * np.pi * month / 12.0)
        darray = DataArray(data, dims=['time'])
        darray.coords['time'] = np.array([datetime(2017, m, 1) for m in month])
        self.darray = darray

    def test_datetime_line_plot(self) -> None:
        self.darray.plot.line()

    def test_datetime_units(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.darray['time'], self.darray)
        assert type(ax.xaxis.get_major_locator()) is mpl.dates.AutoDateLocator

    def test_datetime_plot1d(self) -> None:
        p = self.darray.plot.line()
        ax = p[0].axes
        assert type(ax.xaxis.get_major_locator()) is mpl.dates.AutoDateLocator

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_datetime_plot2d(self) -> None:
        da = DataArray(np.arange(3 * 4).reshape(3, 4), dims=('x', 'y'), coords={'x': [1, 2, 3], 'y': [np.datetime64(f'2000-01-{x:02d}') for x in range(1, 5)]})
        p = da.plot.pcolormesh()
        ax = p.axes
        assert ax is not None
        assert type(ax.xaxis.get_major_locator()) is mpl.dates.AutoDateLocator