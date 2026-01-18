from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
def test_pandas_query_metadata(self):
    hvplot = hvPlotTabular(self.df, query='x>2')
    assert len(hvplot._data) == 2