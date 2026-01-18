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
def test_poly_geom_selection_inverted(self):
    poly = Polygons([[(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)], [(0.25, -0.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)], [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]]).opts(invert_axes=True)
    geom = np.array([(0.2, -0.15), (0.5, 0), (0.75, 0.6), (0.1, 0.6)])
    expr, bbox, region = poly._get_selection_expr_for_stream_value(geometry=geom)
    self.assertEqual(bbox, {'y': np.array([0.2, 0.5, 0.75, 0.1]), 'x': np.array([-0.15, 0, 0.6, 0.6])})
    self.assertEqual(expr.apply(poly, expanded=False), np.array([False, False, True]))
    self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(0.2, -0.15)]]))