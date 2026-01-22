from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class AccessorTestCase(DatasetPropertyTestCase):

    def test_apply_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).apply(lambda c: Scatter(c.select(b=(20, None)).data))
        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).apply(lambda c: Scatter(c.select(b=(20, None)).data))
        self.assertNotEqual(curve, curve2)
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIs(ops[2].output_type, Apply)
        self.assertEqual(ops[2].kwargs, {'mode': None})
        self.assertEqual(ops[3].method_name, '__call__')
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(curve.pipeline(self.ds2), curve2)

    def test_redim_curve(self):
        curve = self.ds.to.curve('a', 'b', groupby=[]).redim.unit(a='kg', b='m')
        curve2 = self.ds2.to.curve('a', 'b', groupby=[]).redim.unit(a='kg', b='m')
        self.assertNotEqual(curve, curve2)
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIs(ops[2].output_type, Redim)
        self.assertEqual(ops[2].kwargs, {'mode': 'dataset'})
        self.assertEqual(ops[3].method_name, '__call__')
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(curve.pipeline(self.ds2), curve2)