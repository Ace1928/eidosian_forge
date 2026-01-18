import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_use_index_disabled_uses_first_cols(self, kind, element):
    plot = self.df.hvplot(use_index=False, kind=kind)
    self.assertEqual(plot.kdims, ['x'])
    self.assertEqual(plot.vdims, ['y'])