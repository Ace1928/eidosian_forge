import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_polygons_string_signature(self):
    polygons = Polygons([], ['a', 'b'])
    self.assertEqual(polygons.kdims, [Dimension('a'), Dimension('b')])