from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
class TestXArrayOverrides(ComparisonTestCase):

    def setUp(self):
        coords = OrderedDict([('time', [0, 1]), ('lat', [0, 1]), ('lon', [0, 1])])
        self.da_img_by_time = xr.DataArray(np.arange(8).reshape((2, 2, 2)), coords, ['time', 'lat', 'lon']).assign_coords(lat1=xr.DataArray([2, 3], dims=['lat']))

    def test_xarray_isel_scalar_metadata(self):
        hvplot = hvPlot(self.da_img_by_time, isel={'time': 1})
        assert hvplot._data.ndim == 2

    def test_xarray_isel_nonscalar_metadata(self):
        hvplot = hvPlot(self.da_img_by_time, isel={'time': [1]})
        assert hvplot._data.ndim == 3
        assert len(hvplot._data.time) == 1

    def test_xarray_sel_metadata(self):
        hvplot = hvPlot(self.da_img_by_time, sel={'time': 1})
        assert hvplot._data.ndim == 2