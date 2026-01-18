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
def test_xarray_dataset_names_and_units(self):
    xs = [0.1, 0.2, 0.3]
    ys = [0, 1]
    zs = np.array([[0, 1], [2, 3], [4, 5]])
    da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name='data_name', dims=['y_dim', 'x_dim'])
    da.attrs['long_name'] = 'data long name'
    da.attrs['units'] = 'array_unit'
    da.x_dim.attrs['units'] = 'x_unit'
    da.y_dim.attrs['long_name'] = 'y axis long name'
    dataset = Dataset(da)
    self.assertEqual(dataset.get_dimension('x_dim'), Dimension('x_dim', unit='x_unit'))
    self.assertEqual(dataset.get_dimension('y_dim'), Dimension('y_dim', label='y axis long name'))
    self.assertEqual(dataset.get_dimension('data_name'), Dimension('data_name', label='data long name', unit='array_unit'))