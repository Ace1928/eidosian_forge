from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class ToTestCase(DatasetPropertyTestCase):

    def test_to_element(self):
        curve = self.ds.to(Curve, 'a', 'b', groupby=[])
        curve2 = self.ds2.to(Curve, 'a', 'b', groupby=[])
        self.assertNotEqual(curve, curve2)
        self.assertEqual(curve.dataset, self.ds)
        scatter = curve.to(Scatter)
        self.assertEqual(scatter.dataset, self.ds)
        ops = curve.pipeline.operations
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertEqual(curve.pipeline(curve.dataset), curve)
        self.assertEqual(curve.pipeline(self.ds2), curve2)

    def test_to_holomap(self):
        curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[v,]
            self.assertEqual(curve.dataset, self.ds)
            self.assertEqual(curve.pipeline(curve.dataset), curve)

    def test_to_holomap_dask(self):
        if dd is None:
            raise SkipTest('Dask required to test .to with dask dataframe.')
        ddf = dd.from_pandas(self.df, npartitions=2)
        dds = Dataset(ddf, kdims=[Dimension('a', label='The a Column'), Dimension('b', label='The b Column'), Dimension('c', label='The c Column'), Dimension('d', label='The d Column')])
        curve_hmap = dds.to(Curve, 'a', 'b', groupby=['c'])
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[v,]
            self.assertEqual(curve.dataset, self.ds)
            self.assertEqual(curve.pipeline(curve.dataset), curve)