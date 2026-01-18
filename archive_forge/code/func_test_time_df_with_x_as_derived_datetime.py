import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_time_df_with_x_as_derived_datetime(self):
    plot = self.time_df.hvplot.scatter(x='time.day', dynamic=False)
    assert list(plot.dimensions()) == ['time.day', 'A']