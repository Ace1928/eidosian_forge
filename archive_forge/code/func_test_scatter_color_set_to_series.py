import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_scatter_color_set_to_series(self):
    if is_dask(self.df['y']):
        y = self.df['y'].compute()
    else:
        y = self.df['y']
    actual = self.df.hvplot.scatter('x', 'y', c=y)
    altered_df = self.df.assign(_color=y)
    expected = altered_df.hvplot.scatter('x', 'y', c='_color')
    self.assertEqual(actual, expected)