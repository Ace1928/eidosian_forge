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
def test_xarray_coord_ordering(self):
    data = np.zeros((3, 4, 5))
    coords = dict([('b', range(3)), ('c', range(4)), ('a', range(5))])
    darray = xr.DataArray(data, coords=coords, dims=['b', 'c', 'a'])
    dataset = xr.Dataset({'value': darray}, coords=coords)
    ds = Dataset(dataset)
    self.assertEqual(ds.kdims, ['b', 'c', 'a'])