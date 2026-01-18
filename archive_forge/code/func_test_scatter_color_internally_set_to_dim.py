import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_scatter_color_internally_set_to_dim(self):
    altered_df = self.cat_df.copy().rename(columns={'category': 'red'})
    plot = altered_df.hvplot.scatter('x', 'y', c='red')
    opts = Store.lookup_options('bokeh', plot, 'style')
    self.assertIsInstance(opts.kwargs['color'], dim)
    self.assertEqual(opts.kwargs['color'].dimension.name, 'red')