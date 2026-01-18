from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@shapelib_available
def test_points_selection_geom(self):
    points = Points([3, 2, 1, 3, 4])
    geom = np.array([(-0.1, -0.1), (1.4, 0), (1.4, 2.2), (-0.1, 2.2)])
    expr, bbox, region = points._get_selection_expr_for_stream_value(geometry=geom)
    self.assertEqual(bbox, {'x': np.array([-0.1, 1.4, 1.4, -0.1]), 'y': np.array([-0.1, 0, 2.2, 2.2])})
    self.assertEqual(expr.apply(points), np.array([False, True, False, False, False]))
    self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.1, -0.1)]]))