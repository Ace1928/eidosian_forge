from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class HistogramTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super().setUp()
        self.hist = self.ds.hist('a', adjoin=False, normed=False)

    def test_construction(self):
        self.assertEqual(self.hist.dataset, self.ds)

    def test_clone(self):
        self.assertEqual(self.hist.clone().dataset, self.ds)

    def test_select_single(self):
        sub_hist = self.hist.select(a=(1, None))
        self.assertEqual(sub_hist.dataset, self.ds)
        ops = sub_hist.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Apply)
        self.assertEqual(ops[2].method_name, '__call__')
        self.assertIsInstance(ops[2].args[0], histogram)
        self.assertEqual(ops[3].method_name, 'select')
        self.assertEqual(ops[3].args, [])
        self.assertEqual(ops[3].kwargs, {'a': (1, None)})
        self.assertEqual(sub_hist.pipeline(sub_hist.dataset), sub_hist)

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

    def test_hist_to_curve(self):
        curve = self.hist.to.curve()
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Apply)
        self.assertEqual(ops[2].method_name, '__call__')
        self.assertIsInstance(ops[2].args[0], histogram)
        self.assertIs(ops[3].output_type, Curve)
        self.assertEqual(curve.pipeline(curve.dataset), curve)