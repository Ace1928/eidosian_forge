import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_wide_chart(self, kind, element):
    plot = self.df.hvplot(kind=kind)
    obj = NdOverlay({'x': element(self.df, 'index', 'x').redim(x='value'), 'y': element(self.df, 'index', 'y').redim(y='value')}, 'Variable')
    self.assertEqual(plot, obj)