from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_violin_single_inverted(self):
    violin = Violin(list(range(10))).opts(invert_axes=True)
    expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(3, 0, 7, 1))
    self.assertEqual(bbox, {'y': (3, 7)})
    self.assertEqual(expr.apply(violin), np.array([False, False, False, True, True, True, True, True, False, False]))
    self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))