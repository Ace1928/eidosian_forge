from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_select_dataset(self):
    ds_select = self.ds.select(b=10)
    ds2_select = self.ds2.select(b=10)
    self.assertNotEqual(ds_select, ds2_select)
    self.assertEqual(ds_select.dataset, self.ds)
    ops = ds_select.pipeline.operations
    self.assertEqual(len(ops), 2)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertEqual(ops[1].method_name, 'select')
    self.assertEqual(ops[1].args, [])
    self.assertEqual(ops[1].kwargs, {'b': 10})
    self.assertEqual(ds_select.pipeline(ds_select.dataset), ds_select)
    self.assertEqual(ds_select.pipeline(self.ds2), ds2_select)