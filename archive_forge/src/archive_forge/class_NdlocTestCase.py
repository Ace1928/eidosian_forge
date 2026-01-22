from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class NdlocTestCase(DatasetPropertyTestCase):

    def setUp(self):
        super().setUp()
        self.ds_grid = Dataset((np.arange(4), np.arange(3), np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])), kdims=['x', 'y'], vdims='z')
        self.ds2_grid = Dataset((np.arange(3), np.arange(3), np.array([[1, 2, 4], [5, 6, 8], [9, 10, 12]])), kdims=['x', 'y'], vdims='z')

    def test_ndloc_dataset(self):
        ds_grid_ndloc = self.ds_grid.ndloc[0:2, 1:3]
        ds2_grid_ndloc = self.ds2_grid.ndloc[0:2, 1:3]
        self.assertNotEqual(ds_grid_ndloc, ds2_grid_ndloc)
        self.assertEqual(ds_grid_ndloc.dataset, self.ds_grid)
        ops = ds_grid_ndloc.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, '_perform_getitem')
        self.assertEqual(ops[1].args, [(slice(0, 2, None), slice(1, 3, None))])
        self.assertEqual(ops[1].kwargs, {})
        self.assertEqual(ds_grid_ndloc.pipeline(ds_grid_ndloc.dataset), ds_grid_ndloc)
        self.assertEqual(ds_grid_ndloc.pipeline(self.ds2_grid), ds2_grid_ndloc)