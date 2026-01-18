from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_reduce_dataset(self):
    ds_reduced = self.ds.reindex(kdims=['b', 'c'], vdims=['a', 'd']).reduce('c', function=np.sum)
    ds2_reduced = self.ds2.reindex(kdims=['b', 'c'], vdims=['a', 'd']).reduce('c', function=np.sum)
    self.assertNotEqual(ds_reduced, ds2_reduced)
    self.assertEqual(ds_reduced.dataset, self.ds)
    self.assertEqual(ds2_reduced.dataset, self.ds2)
    ops = ds_reduced.pipeline.operations
    self.assertEqual(len(ops), 3)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertEqual(ops[1].method_name, 'reindex')
    self.assertEqual(ops[2].method_name, 'reduce')
    self.assertEqual(ops[2].args, ['c'])
    self.assertEqual(ops[2].kwargs, {'function': np.sum})
    self.assertEqual(ds_reduced.pipeline(ds_reduced.dataset), ds_reduced)
    self.assertEqual(ds_reduced.pipeline(self.ds2), ds2_reduced)