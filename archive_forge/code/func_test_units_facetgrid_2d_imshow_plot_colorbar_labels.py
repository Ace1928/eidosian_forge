from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
def test_units_facetgrid_2d_imshow_plot_colorbar_labels(self):
    arr = np.ones((2, 3, 4, 5)) * unit_registry.Pa
    da = xr.DataArray(data=arr, dims=['x', 'y', 'z', 'w'], name='pressure')
    da.plot.imshow(x='x', y='y', col='w')