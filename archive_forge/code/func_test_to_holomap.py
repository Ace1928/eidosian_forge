from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_to_holomap(self):
    curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])
    for v in self.df.c.drop_duplicates():
        curve = curve_hmap.data[v,]
        self.assertEqual(curve.dataset, self.ds)
        self.assertEqual(curve.pipeline(curve.dataset), curve)