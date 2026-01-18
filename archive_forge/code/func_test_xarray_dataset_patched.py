from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_xarray_dataset_patched(self):
    import xarray as xr
    array = np.random.rand(100, 100)
    xr_array = xr.DataArray(array, coords={'x': range(100), 'y': range(100)}, dims=('y', 'x'))
    xr_ds = xr.Dataset({'z': xr_array})
    self.assertIsInstance(xr_ds.hvplot, hvPlot)