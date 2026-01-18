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
def test_select_dropped_dimensions_restoration(self):
    d = np.random.randn(3, 8)
    da = xr.DataArray(d, name='stuff', dims=['chain', 'value'], coords=dict(chain=range(d.shape[0]), value=range(d.shape[1])))
    ds = Dataset(da)
    t = ds.select(chain=0)
    if hasattr(t.data, 'sizes'):
        assert t.data.sizes == dict(chain=1, value=8)
    else:
        assert t.data.dims == dict(chain=1, value=8)
    assert t.data.stuff.shape == (1, 8)