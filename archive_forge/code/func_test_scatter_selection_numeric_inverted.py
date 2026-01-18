from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_scatter_selection_numeric_inverted(self):
    scatter = Scatter([3, 2, 1, 3, 4]).opts(invert_axes=True)
    expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
    self.assertEqual(bbox, {'x': (1, 3)})
    self.assertEqual(expr.apply(scatter), np.array([False, True, True, True, False]))
    self.assertEqual(region, NdOverlay({0: HSpan(1, 3)}))