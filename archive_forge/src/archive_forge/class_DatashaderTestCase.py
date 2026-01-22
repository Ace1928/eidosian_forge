from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
class DatashaderTestCase(DatasetPropertyTestCase):

    def setUp(self):
        if None in (rasterize, datashade, dynspread):
            raise SkipTest('Datashader could not be imported and cannot be tested.')
        super().setUp()

    def test_rasterize_curve(self):
        img = rasterize(self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False)
        img2 = rasterize(self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False)
        self.assertNotEqual(img, img2)
        self.assertEqual(img.dataset, self.ds)
        ops = img.pipeline.operations
        self.assertEqual(len(ops), 3)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIsInstance(ops[2], rasterize)
        self.assertEqual(img.pipeline(img.dataset), img)
        self.assertEqual(img.pipeline(self.ds2), img2)

    def test_datashade_curve(self):
        rgb = dynspread(datashade(self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False), dynamic=False)
        rgb2 = dynspread(datashade(self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False), dynamic=False)
        self.assertNotEqual(rgb, rgb2)
        self.assertEqual(rgb.dataset, self.ds)
        ops = rgb.pipeline.operations
        self.assertEqual(len(ops), 4)
        self.assertIs(ops[0].output_type, Dataset)
        self.assertIs(ops[1].output_type, Curve)
        self.assertIsInstance(ops[2], datashade)
        self.assertIsInstance(ops[3], dynspread)
        self.assertEqual(rgb.pipeline(rgb.dataset), rgb)
        self.assertEqual(rgb.pipeline(self.ds2), rgb2)