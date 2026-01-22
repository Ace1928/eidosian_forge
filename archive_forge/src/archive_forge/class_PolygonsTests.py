import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
class PolygonsTests(ComparisonTestCase):

    def setUp(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        self.single_poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
        self.multi_poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        self.multi_poly_no_hole = Polygons([{'x': xs, 'y': ys}])
        self.distinct_polys = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'value': 0}, {'x': [4, 6, 6], 'y': [0, 2, 1], 'value': 1}], vdims='value')

    def test_single_poly_holes_match(self):
        self.assertTrue(self.single_poly.interface.has_holes(self.single_poly))
        paths = self.single_poly.split(datatype='array')
        holes = self.single_poly.interface.holes(self.single_poly)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 1)
        self.assertEqual(len(holes[0][0]), 2)

    def test_multi_poly_holes_match(self):
        self.assertTrue(self.multi_poly.interface.has_holes(self.multi_poly))
        paths = self.multi_poly.split(datatype='array')
        holes = self.multi_poly.interface.holes(self.multi_poly)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 2)
        self.assertEqual(len(holes[0][1]), 0)

    def test_multi_poly_empty_holes(self):
        poly = Polygons([])
        self.assertFalse(poly.interface.has_holes(poly))
        self.assertEqual(poly.interface.holes(poly), [])

    def test_multi_poly_no_holes_match(self):
        self.assertFalse(self.multi_poly_no_hole.interface.has_holes(self.multi_poly_no_hole))
        paths = self.multi_poly_no_hole.split(datatype='array')
        holes = self.multi_poly_no_hole.interface.holes(self.multi_poly_no_hole)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 0)
        self.assertEqual(len(holes[0][1]), 0)

    def test_distinct_multi_poly_holes_match(self):
        self.assertTrue(self.distinct_polys.interface.has_holes(self.distinct_polys))
        paths = self.distinct_polys.split(datatype='array')
        holes = self.distinct_polys.interface.holes(self.distinct_polys)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 2)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 2)
        self.assertEqual(len(holes[0][1]), 0)
        self.assertEqual(len(holes[1]), 1)
        self.assertEqual(len(holes[1][0]), 0)

    def test_single_poly_hole_validation(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        with self.assertRaises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[], []]}])

    def test_multi_poly_hole_validation(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        with self.assertRaises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[]]}])