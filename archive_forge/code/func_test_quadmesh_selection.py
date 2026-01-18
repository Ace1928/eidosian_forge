from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
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