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
class TestAxesKwargs:

    @pytest.fixture(params=[1, 2, 3])
    def data_array(self, request):
        """
        Return a simple DataArray
        """
        dims = request.param
        if dims == 1:
            return DataArray(easy_array((10,)))
        if dims == 2:
            return DataArray(easy_array((10, 3)))
        if dims == 3:
            return DataArray(easy_array((10, 3, 2)))

    @pytest.fixture(params=[1, 2])
    def data_array_logspaced(self, request):
        """
        Return a simple DataArray with logspaced coordinates
        """
        dims = request.param
        if dims == 1:
            return DataArray(np.arange(7), dims=('x',), coords={'x': np.logspace(-3, 3, 7)})
        if dims == 2:
            return DataArray(np.arange(16).reshape(4, 4), dims=('y', 'x'), coords={'x': np.logspace(-1, 2, 4), 'y': np.logspace(-5, -1, 4)})

    @pytest.mark.parametrize('xincrease', [True, False])
    def test_xincrease_kwarg(self, data_array, xincrease) -> None:
        with figure_context():
            data_array.plot(xincrease=xincrease)
            assert plt.gca().xaxis_inverted() == (not xincrease)

    @pytest.mark.parametrize('yincrease', [True, False])
    def test_yincrease_kwarg(self, data_array, yincrease) -> None:
        with figure_context():
            data_array.plot(yincrease=yincrease)
            assert plt.gca().yaxis_inverted() == (not yincrease)

    @pytest.mark.parametrize('xscale', ['linear', 'logit', 'symlog'])
    def test_xscale_kwarg(self, data_array, xscale) -> None:
        with figure_context():
            data_array.plot(xscale=xscale)
            assert plt.gca().get_xscale() == xscale

    @pytest.mark.parametrize('yscale', ['linear', 'logit', 'symlog'])
    def test_yscale_kwarg(self, data_array, yscale) -> None:
        with figure_context():
            data_array.plot(yscale=yscale)
            assert plt.gca().get_yscale() == yscale

    def test_xscale_log_kwarg(self, data_array_logspaced) -> None:
        xscale = 'log'
        with figure_context():
            data_array_logspaced.plot(xscale=xscale)
            assert plt.gca().get_xscale() == xscale

    def test_yscale_log_kwarg(self, data_array_logspaced) -> None:
        yscale = 'log'
        with figure_context():
            data_array_logspaced.plot(yscale=yscale)
            assert plt.gca().get_yscale() == yscale

    def test_xlim_kwarg(self, data_array) -> None:
        with figure_context():
            expected = (0.0, 1000.0)
            data_array.plot(xlim=[0, 1000])
            assert plt.gca().get_xlim() == expected

    def test_ylim_kwarg(self, data_array) -> None:
        with figure_context():
            data_array.plot(ylim=[0, 1000])
            expected = (0.0, 1000.0)
            assert plt.gca().get_ylim() == expected

    def test_xticks_kwarg(self, data_array) -> None:
        with figure_context():
            data_array.plot(xticks=np.arange(5))
            expected = np.arange(5).tolist()
            assert_array_equal(plt.gca().get_xticks(), expected)

    def test_yticks_kwarg(self, data_array) -> None:
        with figure_context():
            data_array.plot(yticks=np.arange(5))
            expected = np.arange(5)
            assert_array_equal(plt.gca().get_yticks(), expected)