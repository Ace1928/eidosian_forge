import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_scatter_color_by_legend_position(self):
    plot = self.cat_df.hvplot.scatter('x', 'y', c='category', legend='left')
    opts = Store.lookup_options('bokeh', plot, 'plot')
    self.assertEqual(opts.kwargs['legend_position'], 'left')