from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
class SpatialPandasTest(GeomTests, RoundTripTests):
    """
    Test of the SpatialPandasInterface.
    """
    datatype = 'spatialpandas'
    interface = SpatialPandasInterface
    __test__ = True

    def setUp(self):
        if spatialpandas is None:
            raise SkipTest('SpatialPandasInterface requires spatialpandas, skipping tests')
        super(GeomTests, self).setUp()

    def test_array_points_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1 + i, i), (2 + i, i), (3 + i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.iloc[3, 0]

    def test_point_constructor(self):
        points = Points([{'x': 0, 'y': 1}, {'x': 1, 'y': 0}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(points.data.geometry.dtype, PointDtype)
        self.assertEqual(points.data.iloc[0, 0].flat_values, np.array([0, 1]))
        self.assertEqual(points.data.iloc[1, 0].flat_values, np.array([1, 0]))

    def test_multi_point_constructor(self):
        xs = [1, 2, 3, 2]
        ys = [2, 0, 7, 4]
        points = Points([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(points.data.geometry.dtype, MultiPointDtype)
        self.assertEqual(points.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7, 2, 4]))
        self.assertEqual(points.data.iloc[1, 0].buffer_values, np.array([2, 4, 3, 7, 2, 0, 1, 2]))

    def test_line_constructor(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(path.data.geometry.dtype, LineDtype)
        self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7]))
        self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 7, 2, 0, 1, 2]))

    def test_multi_line_constructor(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(path.data.geometry.dtype, MultiLineDtype)
        self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7, 6, 7, 7, 5, 3, 2]))
        self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 2, 7, 5, 6, 7, 3, 7, 2, 0, 1, 2]))

    def test_polygon_constructor(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        path = Polygons([{'x': xs, 'y': ys, 'holes': holes}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(path.data.geometry.dtype, PolygonDtype)
        self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1.0, 2.0, 2.0, 0.0, 3.0, 7.0, 1.0, 2.0, 1.5, 2.0, 2.0, 3.0, 1.6, 1.6, 1.5, 2.0, 2.1, 4.5, 2.5, 5.0, 2.3, 3.5, 2.1, 4.5]))
        self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 7, 1, 2, 2, 0, 3, 7]))

    def test_multi_polygon_constructor(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
        path = Polygons([{'x': xs, 'y': ys, 'holes': holes}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
        self.assertIsInstance(path.data.geometry.dtype, MultiPolygonDtype)
        self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1.0, 2.0, 2.0, 0.0, 3.0, 7.0, 1.0, 2.0, 1.5, 2.0, 2.0, 3.0, 1.6, 1.6, 1.5, 2.0, 2.1, 4.5, 2.5, 5.0, 2.3, 3.5, 2.1, 4.5, 6.0, 7.0, 3.0, 2.0, 7.0, 5.0, 6.0, 7.0]))
        self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 2, 7, 5, 6, 7, 3, 2, 3, 7, 1, 2, 2, 0, 3, 7]))

    def test_geometry_array_constructor(self):
        polygons = MultiPolygonArray([[[[0, 0, 1, 0, 2, 2, -1, 4, 0, 0], [0.5, 1, 1, 2, 1.5, 1.5, 0.5, 1], [0, 2, 0, 2.5, 0.5, 2.5, 0.5, 2, 0, 2]], [[-0.5, 3, 1.5, 3, 1.5, 4, -0.5, 3]]], [[[1.25, 0, 1.25, 2, 4, 2, 4, 0, 1.25, 0], [1.5, 0.25, 3.75, 0.25, 3.75, 1.75, 1.5, 1.75, 1.5, 0.25]]]])
        path = Polygons(polygons)
        self.assertIsInstance(path.data.geometry.dtype, MultiPolygonDtype)