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
class TestPcolormeshLogscale(PlotTestCase):
    """
    Test pcolormesh axes when x and y are in logscale
    """
    plotfunc = staticmethod(xplt.pcolormesh)

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.boundaries = (-1, 9, -4, 3)
        shape = (8, 11)
        x = np.logspace(self.boundaries[0], self.boundaries[1], shape[1])
        y = np.logspace(self.boundaries[2], self.boundaries[3], shape[0])
        da = DataArray(easy_array(shape, start=-1), dims=['y', 'x'], coords={'y': y, 'x': x}, name='testvar')
        self.darray = da

    def test_interval_breaks_logspace(self) -> None:
        """
        Check if the outer vertices of the pcolormesh are the expected values

        Checks bugfix for #5333
        """
        artist = self.darray.plot.pcolormesh(xscale='log', yscale='log')
        x_vertices = [p.vertices[:, 0] for p in artist.properties()['paths']]
        y_vertices = [p.vertices[:, 1] for p in artist.properties()['paths']]
        xmin, xmax = (np.min(x_vertices), np.max(x_vertices))
        ymin, ymax = (np.min(y_vertices), np.max(y_vertices))
        log_interval = 0.5
        np.testing.assert_allclose(xmin, 10 ** (self.boundaries[0] - log_interval))
        np.testing.assert_allclose(xmax, 10 ** (self.boundaries[1] + log_interval))
        np.testing.assert_allclose(ymin, 10 ** (self.boundaries[2] - log_interval))
        np.testing.assert_allclose(ymax, 10 ** (self.boundaries[3] + log_interval))