from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class IlocTestCase(DatasetPropertyTestCase):

    def test_iloc_dataset(self):
        ds_iloc = self.ds.iloc[[0, 2]]
        ds2_iloc = self.ds2.iloc[[0, 2]]
        self.assertNotEqual(ds_iloc, ds2_iloc)
        self.assertEqual(ds_iloc.dataset, self.ds)
        ops = ds_iloc.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, '_perform_getitem')
        self.assertEqual(ops[1].args, [[0, 2]])
        self.assertEqual(ops[1].kwargs, {})
        self.assertEqual(ds_iloc.pipeline(ds_iloc.dataset), ds_iloc)
        self.assertEqual(ds_iloc.pipeline(self.ds2), ds2_iloc)

    def test_iloc_curve(self):
        curve_iloc = self.ds.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]
        curve2_iloc = self.ds2.to.curve('a', 'b', groupby=[]).iloc[[0, 2]]
        self.assertNotEqual(curve_iloc, curve2_iloc)
        self.assertEqual(curve_iloc.dataset, self.ds)
        ops = curve_iloc.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, '_perform_getitem')
        self.assertEqual(ops[2].args, [[0, 2]])
        self.assertEqual(ops[2].kwargs, {})
        self.assertEqual(curve_iloc.pipeline(curve_iloc.dataset), curve_iloc)
        self.assertEqual(curve_iloc.pipeline(self.ds2), curve2_iloc)