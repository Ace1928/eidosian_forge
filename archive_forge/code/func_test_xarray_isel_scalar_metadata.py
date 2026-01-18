from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
def test_xarray_isel_scalar_metadata(self):
    hvplot = hvPlot(self.da_img_by_time, isel={'time': 1})
    assert hvplot._data.ndim == 2