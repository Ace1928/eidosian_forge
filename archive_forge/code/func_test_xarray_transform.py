from unittest import SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
from holoviews.element.comparison import ComparisonTestCase
def test_xarray_transform(self):
    import xarray as xr
    data = np.arange(0, 60).reshape(6, 10)
    x = np.arange(10)
    y = np.arange(6)
    da = xr.DataArray(data, coords={'y': y, 'x': x}, dims=('y', 'x'), name='value')
    img = da.hvplot.image(transforms=dict(value=hv.dim('value') * 10))
    self.assertEqual(img.data.value.data, da.data * 10)