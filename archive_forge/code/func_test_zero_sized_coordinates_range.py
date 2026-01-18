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
def test_zero_sized_coordinates_range(self):
    da = xr.DataArray(np.empty((2, 0)), dims=('y', 'x'), coords={'x': [], 'y': [0, 1]}, name='A')
    ds = Dataset(da)
    x0, x1 = ds.range('x')
    self.assertTrue(np.isnan(x0))
    self.assertTrue(np.isnan(x1))
    z0, z1 = ds.range('A')
    self.assertTrue(np.isnan(z0))
    self.assertTrue(np.isnan(z1))