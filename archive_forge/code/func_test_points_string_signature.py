import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_points_string_signature(self):
    points = Points([], ['a', 'b'], 'c')
    self.assertEqual(points.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(points.vdims, [Dimension('c')])