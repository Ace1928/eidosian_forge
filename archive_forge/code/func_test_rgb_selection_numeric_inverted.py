from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_rgb_selection_numeric_inverted(self):
    img = RGB(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3, 3))).opts(invert_axes=True)
    expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(1.5, 0.5, 3.1, 2.1))
    self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
    self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[False, False, False], [False, False, False], [False, True, True], [False, True, True]]))
    self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]) * Path([]))