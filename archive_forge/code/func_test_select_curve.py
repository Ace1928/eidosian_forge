from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_select_curve(self):
    curve_select = self.ds.to.curve('a', 'b', groupby=[]).select(b=10)
    curve2_select = self.ds2.to.curve('a', 'b', groupby=[]).select(b=10)
    self.assertNotEqual(curve_select, curve2_select)
    self.assertEqual(curve_select.dataset, self.ds)
    ops = curve_select.pipeline.operations
    self.assertEqual(len(ops), 3)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertEqual(ops[2].method_name, 'select')
    self.assertEqual(ops[2].args, [])
    self.assertEqual(ops[2].kwargs, {'b': 10})
    self.assertEqual(curve_select.pipeline(curve_select.dataset), curve_select)
    self.assertEqual(curve_select.pipeline(self.ds2), curve2_select)