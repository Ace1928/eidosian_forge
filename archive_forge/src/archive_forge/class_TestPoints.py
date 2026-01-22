import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
class TestPoints(ComparisonTestCase):

    def test_empty_geom_conversion(self):
        points = Points([])
        self.assertEqual(points.geom(), GeometryCollection())

    def test_single_geom_conversion(self):
        points = Points([(0, 0)])
        geom = points.geom()
        self.assertIsInstance(geom, Point)
        self.assertEqual(np.column_stack(geom.xy), np.array([[0, 0]]))

    def test_multi_geom_conversion(self):
        points = Points([(0, 0), (1, 2.5)])
        geom = points.geom()
        self.assertIsInstance(geom, MultiPoint)
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(np.column_stack(geom.geoms[0].xy), np.array([[0, 0]]))
        self.assertEqual(np.column_stack(geom.geoms[1].xy), np.array([[1, 2.5]]))