from __future__ import annotations
import pytest
from dask.context import config
from os import path
from itertools import product
import datashader as ds
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
@pytest.mark.parametrize('in_size, out_size, agg', product(range(5, 8), range(2, 5), ['mean', 'min', 'max', 'first', 'last', 'var', 'std', 'mode']))
def test_raster_distributed_downsample(in_size, out_size, agg):
    """
    Ensure that distributed regrid is equivalent to regular regrid.
    """
    cvs = ds.Canvas(plot_height=out_size, plot_width=out_size)
    vs = np.linspace(-1, 1, in_size)
    xs, ys = np.meshgrid(vs, vs)
    arr = np.sin(xs * ys)
    darr = da.from_array(arr, (2, 2))
    coords = [('y', range(in_size)), ('x', range(in_size))]
    xr_darr = xr.DataArray(darr, coords=coords, name='z')
    xr_arr = xr.DataArray(arr, coords=coords, name='z')
    agg_arr = cvs.raster(xr_arr, agg=agg)
    agg_darr = cvs.raster(xr_darr, agg=agg)
    assert np.allclose(agg_arr.data, agg_darr.data.compute())
    assert np.allclose(agg_arr.x.values, agg_darr.x.values)
    assert np.allclose(agg_arr.y.values, agg_darr.y.values)