from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class ReindexTestCase(DatasetPropertyTestCase):

    def test_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
        ds2_ab = self.ds2.reindex(kdims=['a'], vdims=['b'])
        self.assertNotEqual(ds_ab, ds2_ab)
        self.assertEqual(ds_ab.dataset, self.ds)
        ops = ds_ab.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[1].args, [])
        self.assertEqual(ops[1].kwargs, dict(kdims=['a'], vdims=['b']))
        self.assertEqual(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        self.assertEqual(ds_ab.pipeline(self.ds2), ds2_ab)

    def test_double_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b', 'c']).reindex(kdims=['a'], vdims=['b'])
        ds2_ab = self.ds2.reindex(kdims=['a'], vdims=['b', 'c']).reindex(kdims=['a'], vdims=['b'])
        self.assertNotEqual(ds_ab, ds2_ab)
        self.assertEqual(ds_ab.dataset, self.ds)
        ops = ds_ab.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertEqual(ops[1].method_name, 'reindex')
        self.assertEqual(ops[1].args, [])
        self.assertEqual(ops[1].kwargs, dict(kdims=['a'], vdims=['b', 'c']))
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims=['a'], vdims=['b']))
        self.assertEqual(ds_ab.pipeline(ds_ab.dataset), ds_ab)
        self.assertEqual(ds_ab.pipeline(self.ds2), ds2_ab)

    def test_reindex_curve(self):
        curve_ba = self.ds.to(Curve, 'a', 'b', groupby=[]).reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(Curve, 'a', 'b', groupby=[]).reindex(kdims='b', vdims='a')
        self.assertNotEqual(curve_ba, curve2_ba)
        self.assertEqual(curve_ba.dataset, self.ds)
        ops = curve_ba.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims='b', vdims='a'))
        self.assertEqual(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        self.assertEqual(curve_ba.pipeline(self.ds2), curve2_ba)

    def test_double_reindex_curve(self):
        curve_ba = self.ds.to(Curve, 'a', ['b', 'c'], groupby=[]).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')
        curve2_ba = self.ds2.to(Curve, 'a', ['b', 'c'], groupby=[]).reindex(kdims='a', vdims='b').reindex(kdims='b', vdims='a')
        self.assertNotEqual(curve_ba, curve2_ba)
        self.assertEqual(curve_ba.dataset, self.ds)
        ops = curve_ba.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(ops[2].method_name, 'reindex')
        self.assertEqual(ops[2].args, [])
        self.assertEqual(ops[2].kwargs, dict(kdims='a', vdims='b'))
        self.assertEqual(ops[3].method_name, 'reindex')
        self.assertEqual(ops[3].args, [])
        self.assertEqual(ops[3].kwargs, dict(kdims='b', vdims='a'))
        self.assertEqual(curve_ba.pipeline(curve_ba.dataset), curve_ba)
        self.assertEqual(curve_ba.pipeline(self.ds2), curve2_ba)