from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
class TestSelection2DExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_points_selection_numeric(self):
        points = Points([3, 2, 1, 3, 4])
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(1, 0, 3, 2)]) * Path([]))

    def test_points_selection_numeric_inverted(self):
        points = Points([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]) * Path([]))

    @shapelib_available
    def test_points_selection_geom(self):
        points = Points([3, 2, 1, 3, 4])
        geom = np.array([(-0.1, -0.1), (1.4, 0), (1.4, 2.2), (-0.1, 2.2)])
        expr, bbox, region = points._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x': np.array([-0.1, 1.4, 1.4, -0.1]), 'y': np.array([-0.1, 0, 2.2, 2.2])})
        self.assertEqual(expr.apply(points), np.array([False, True, False, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.1, -0.1)]]))

    @shapelib_available
    def test_points_selection_geom_inverted(self):
        points = Points([3, 2, 1, 3, 4]).opts(invert_axes=True)
        geom = np.array([(-0.1, -0.1), (1.4, 0), (1.4, 2.2), (-0.1, 2.2)])
        expr, bbox, region = points._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y': np.array([-0.1, 1.4, 1.4, -0.1]), 'x': np.array([-0.1, 0, 2.2, 2.2])})
        self.assertEqual(expr.apply(points), np.array([False, False, True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.1, -0.1)]]))

    def test_points_selection_categorical(self):
        points = Points((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None)
        self.assertEqual(bbox, {'x': ['B', 'A', 'C'], 'y': (1, 3)})
        self.assertEqual(expr.apply(points), np.array([True, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]) * Path([]))

    def test_points_selection_numeric_index_cols(self):
        points = Points([3, 2, 1, 3, 2])
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2), index_cols=['y'])
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, False, True, False, False]))
        self.assertEqual(region, None)

    def test_scatter_selection_numeric(self):
        scatter = Scatter([3, 2, 1, 3, 4])
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_scatter_selection_numeric_inverted(self):
        scatter = Scatter([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 3)}))

    def test_scatter_selection_categorical(self):
        scatter = Scatter((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None)
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(scatter), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_scatter_selection_numeric_index_cols(self):
        scatter = Scatter([3, 2, 1, 3, 2])
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2), index_cols=['y'])
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_image_selection_numeric(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3)))
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(0.5, 1.5, 2.1, 3.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[False, False, False], [False, False, False], [False, True, True], [False, True, True]]))
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]) * Path([]))

    def test_image_selection_numeric_inverted(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3))).opts(invert_axes=True)
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(1.5, 0.5, 3.1, 2.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[False, False, False], [False, False, False], [False, True, True], [False, True, True]]))
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]) * Path([]))

    @ds_available
    @spd_available
    def test_img_selection_geom(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3)))
        geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
        expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x': np.array([-0.4, 0.6, 0.4, -0.1]), 'y': np.array([-0.1, -0.1, 1.7, 1.7])})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[1.0, np.nan, np.nan], [1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))

    @ds_available
    def test_img_selection_geom_inverted(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3))).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
        expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y': np.array([-0.4, 0.6, 0.4, -0.1]), 'x': np.array([-0.1, -0.1, 1.7, 1.7])})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[True, True, False], [False, False, False], [False, False, False], [False, False, False]]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))

    def test_rgb_selection_numeric(self):
        img = RGB(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3, 3)))
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(0.5, 1.5, 2.1, 3.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[False, False, False], [False, False, False], [False, True, True], [False, True, True]]))
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]) * Path([]))

    def test_rgb_selection_numeric_inverted(self):
        img = RGB(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3, 3))).opts(invert_axes=True)
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(1.5, 0.5, 3.1, 2.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[False, False, False], [False, False, False], [False, True, True], [False, True, True]]))
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]) * Path([]))

    def test_quadmesh_selection(self):
        n = 4
        coords = np.linspace(-1.5, 1.5, n)
        X, Y = np.meshgrid(coords, coords)
        Qx = np.cos(Y) - np.cos(X)
        Qy = np.sin(Y) + np.sin(X)
        Z = np.sqrt(X ** 2 + Y ** 2)
        qmesh = QuadMesh((Qx, Qy, Z))
        expr, bbox, region = qmesh._get_selection_expr_for_stream_value(bounds=(0, -0.5, 0.7, 1.5))
        self.assertEqual(bbox, {'x': (0, 0.7), 'y': (-0.5, 1.5)})
        self.assertEqual(expr.apply(qmesh, expanded=True, flat=False), np.array([[False, False, False, True], [False, False, True, False], [False, True, True, False], [True, False, False, False]]))
        self.assertEqual(region, Rectangles([(0, -0.5, 0.7, 1.5)]) * Path([]))

    def test_quadmesh_selection_inverted(self):
        n = 4
        coords = np.linspace(-1.5, 1.5, n)
        X, Y = np.meshgrid(coords, coords)
        Qx = np.cos(Y) - np.cos(X)
        Qy = np.sin(Y) + np.sin(X)
        Z = np.sqrt(X ** 2 + Y ** 2)
        qmesh = QuadMesh((Qx, Qy, Z)).opts(invert_axes=True)
        expr, bbox, region = qmesh._get_selection_expr_for_stream_value(bounds=(0, -0.5, 0.7, 1.5))
        self.assertEqual(bbox, {'x': (-0.5, 1.5), 'y': (0, 0.7)})
        self.assertEqual(expr.apply(qmesh, expanded=True, flat=False), np.array([[False, False, False, True], [False, False, True, True], [False, True, False, False], [True, False, False, False]]))
        self.assertEqual(region, Rectangles([(0, -0.5, 0.7, 1.5)]) * Path([]))