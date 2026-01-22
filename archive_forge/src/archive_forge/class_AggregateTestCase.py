from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class AggregateTestCase(DatasetPropertyTestCase):

    def test_aggregate_dataset(self):
        ds_aggregated = self.ds.reindex(kdims=['b', 'c'], vdims=['a', 'd']).aggregate('b', function=np.sum)
        ds2_aggregated = self.ds2.reindex(kdims=['b', 'c'], vdims=['a', 'd']).aggregate('b', function=np.sum)
        self.assertNotEqual(ds_aggregated, ds2_aggregated)
        self.assertEqual(ds_aggregated.dataset, self.ds)
        self.assertEqual(ds2_aggregated.dataset, self.ds2)
        ops = ds_aggregated.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[2].method_name, 'aggregate')
        self.assertEqual(ops[2].args, ['b'])
        self.assertEqual(ops[2].kwargs, {'function': np.sum})
        self.assertEqual(ds_aggregated.pipeline(ds_aggregated.dataset), ds_aggregated)
        self.assertEqual(ds_aggregated.pipeline(self.ds2), ds2_aggregated)