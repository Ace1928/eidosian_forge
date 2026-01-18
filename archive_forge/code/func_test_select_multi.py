from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_select_multi(self):
    sub_hist = self.hist.select(a=(1, None), b=100)
    self.assertNotEqual(sub_hist.dataset, self.ds.select(a=(1, None), b=100))
    self.assertEqual(sub_hist.dataset, self.ds)
    ops = sub_hist.pipeline.operations
    self.assertEqual(len(ops), 4)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Apply)
    self.assertEqual(ops[2].method_name, '__call__')
    self.assertIsInstance(ops[2].args[0], histogram)
    self.assertEqual(ops[3].method_name, 'select')
    self.assertEqual(ops[3].args, [])
    self.assertEqual(ops[3].kwargs, {'a': (1, None), 'b': 100})
    self.assertEqual(sub_hist.pipeline(sub_hist.dataset), sub_hist)