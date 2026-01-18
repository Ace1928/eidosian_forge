from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_to_element(self):
    curve = self.ds.to(Curve, 'a', 'b', groupby=[])
    curve2 = self.ds2.to(Curve, 'a', 'b', groupby=[])
    self.assertNotEqual(curve, curve2)
    self.assertEqual(curve.dataset, self.ds)
    scatter = curve.to(Scatter)
    self.assertEqual(scatter.dataset, self.ds)
    ops = curve.pipeline.operations
    self.assertEqual(len(ops), 2)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertEqual(curve.pipeline(curve.dataset), curve)
    self.assertEqual(curve.pipeline(self.ds2), curve2)