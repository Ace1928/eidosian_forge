import datetime as dt
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, XArrayInterface, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import HSV, RGB, Image, ImageStack, QuadMesh
from .test_gridinterface import BaseGridInterfaceTests
from .test_imageinterface import (
def test_xarray_dataset_dataarray_vs_dataset(self):
    xs = [0.1, 0.2, 0.3]
    ys = [0, 1]
    zs = np.array([[0, 1], [2, 3], [4, 5]])
    da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name='data_name', dims=['y_dim', 'x_dim'])
    da.attrs['long_name'] = 'data long name'
    da.attrs['units'] = 'array_unit'
    da.x_dim.attrs['units'] = 'x_unit'
    da.y_dim.attrs['long_name'] = 'y axis long name'
    ds = da.to_dataset()
    dataset_from_da = Dataset(da)
    dataset_from_ds = Dataset(ds)
    self.assertEqual(dataset_from_da, dataset_from_ds)
    da_rev = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name='data_name', dims=['x_dim', 'y_dim'])
    da_rev.attrs['long_name'] = 'data long name'
    da_rev.attrs['units'] = 'array_unit'
    da_rev.x_dim.attrs['units'] = 'x_unit'
    da_rev.y_dim.attrs['long_name'] = 'y axis long name'
    ds_rev = da_rev.to_dataset()
    dataset_from_da_rev = Dataset(da_rev)
    dataset_from_ds_rev = Dataset(ds_rev)
    self.assertEqual(dataset_from_da_rev, dataset_from_ds_rev)