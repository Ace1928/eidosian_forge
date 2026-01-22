from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
class TestDatashader(TestCase):

    def setUp(self):
        try:
            import datashader
        except:
            raise SkipTest('Datashader not available')
        if sys.maxsize < 2 ** 32:
            raise SkipTest('Datashader does not support 32-bit systems')
        self.df = pd.DataFrame(np.random.randn(1000, 3), columns=['a', 'b', 'c'])

    def test_rasterize_datashade_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            scatter_matrix(self.df, rasterize=True, datashade=True)

    def test_spread_but_no_rasterize_or_datashade(self):
        with self.assertRaises(ValueError):
            scatter_matrix(self.df, dynspread=True)
        with self.assertRaises(ValueError):
            scatter_matrix(self.df, spread=True)
        with self.assertRaises(ValueError):
            scatter_matrix(self.df, dynspread=True, spread=True)

    @parameterized.expand([('rasterize',), ('datashade',)])
    def test_rasterization(self, operation):
        sm = scatter_matrix(self.df, **{operation: True})
        dm = sm['a', 'b']
        self.assertEqual(dm.callback.operation.name, operation)
        dm[()]
        self.assertEqual(len(dm.last.pipeline.operations), 3)

    @parameterized.expand([('rasterize',), ('datashade',)])
    def test_datashade_aggregator(self, operation):
        sm = scatter_matrix(self.df, aggregator='mean', **{operation: True})
        dm = sm['a', 'b']
        dm[()]
        self.assertEqual(dm.last.pipeline.operations[-1].aggregator, 'mean')

    @parameterized.expand([('spread',), ('dynspread',)])
    def test_spread_rasterize(self, operation):
        sm = scatter_matrix(self.df, rasterize=True, **{operation: True})
        dm = sm['a', 'b']
        dm[()]
        self.assertEqual(len(dm.last.pipeline.operations), 4)

    @parameterized.expand([('spread',), ('dynspread',)])
    def test_spread_datashade(self, operation):
        sm = scatter_matrix(self.df, datashade=True, **{operation: True})
        dm = sm['a', 'b']
        dm[()]
        self.assertEqual(len(dm.last.pipeline.operations), 4)

    @parameterized.expand([('spread',), ('dynspread',)])
    def test_spread_kwargs(self, operation):
        sm = scatter_matrix(self.df, datashade=True, **{operation: True, 'shape': 'circle'})
        dm = sm['a', 'b']
        dm[()]
        self.assertEqual(dm.last.pipeline.operations[-1].args[0].keywords['shape'], 'circle')