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
def test_units_in_2d_plot_colorbar_label(self):
    arr = np.ones((2, 3)) * unit_registry.Pa
    da = xr.DataArray(data=arr, dims=['x', 'y'], name='pressure')
    fig, (ax, cax) = plt.subplots(1, 2)
    ax = da.plot.contourf(ax=ax, cbar_ax=cax, add_colorbar=True)
    assert cax.get_ylabel() == 'pressure [pascal]'