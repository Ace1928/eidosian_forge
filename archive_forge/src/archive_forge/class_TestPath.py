import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
class TestPath(ComparisonTestCase):

    def test_empty_geom_conversion(self):
        path = Path([])
        self.assertEqual(path.geom(), GeometryCollection())

    def test_single_geom_conversion(self):
        path = Path([[(0, 0), (1, 1), (2, 0)]])
        geom = path.geom()
        self.assertIsInstance(geom, LineString)
        self.assertEqual(np.array(geom.coords), np.array([[0, 0], [1, 1], [2, 0]]))

    def test_multi_geom_conversion(self):
        path = Path([[(0, 0), (1, 1), (2, 0)], [(3, 2), (2.5, 1.5)]])
        geom = path.geom()
        self.assertIsInstance(geom, MultiLineString)
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(np.array(geom.geoms[0].coords), np.array([[0, 0], [1, 1], [2, 0]]))
        self.assertEqual(np.array(geom.geoms[1].coords), np.array([[3, 2], [2.5, 1.5]]))