import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_single_poly_holes_match(self):
    self.assertTrue(self.single_poly.interface.has_holes(self.single_poly))
    paths = self.single_poly.split(datatype='array')
    holes = self.single_poly.interface.holes(self.single_poly)
    self.assertEqual(len(paths), len(holes))
    self.assertEqual(len(holes), 1)
    self.assertEqual(len(holes[0]), 1)
    self.assertEqual(len(holes[0][0]), 2)