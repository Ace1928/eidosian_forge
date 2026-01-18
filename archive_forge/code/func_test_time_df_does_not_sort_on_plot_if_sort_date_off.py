import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_time_df_does_not_sort_on_plot_if_sort_date_off(self):
    scrambled = self.time_df.sample(frac=1)
    plot = scrambled.hvplot(x='time', sort_date=False)
    assert (plot.data == scrambled).all().all()
    assert len(plot.data.time.unique()) == len(plot.data.time)