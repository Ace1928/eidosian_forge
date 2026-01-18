from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@shapely_available
def test_rect_geom_selection_inverted(self):
    rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
    geom = np.array([(-0.4, -0.1), (3.2, -0.1), (3.2, 4.1), (-0.1, 4.2)])
    expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
    self.assertEqual(bbox, {'y0': np.array([-0.4, 3.2, 3.2, -0.1]), 'x0': np.array([-0.1, -0.1, 4.1, 4.2]), 'y1': np.array([-0.4, 3.2, 3.2, -0.1]), 'x1': np.array([-0.1, -0.1, 4.1, 4.2])})
    self.assertEqual(expr.apply(rect), np.array([True, False, False]))
    self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))