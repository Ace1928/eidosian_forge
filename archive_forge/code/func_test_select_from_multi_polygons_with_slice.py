import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_select_from_multi_polygons_with_slice(self):
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
    poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1}, {'x': xs[::-1], 'y': ys[::-1], 'z': 2}, {'x': xs[:3], 'y': ys[:3], 'z': 3}], ['x', 'y'], 'z', datatype=[self.datatype])
    expected = Polygons([{'x': xs[::-1], 'y': ys[::-1], 'z': 2}, {'x': xs[:3], 'y': ys[:3], 'z': 3}], ['x', 'y'], 'z', datatype=[self.datatype])
    self.assertIs(poly.interface, self.interface)
    self.assertEqual(poly.select(z=(2, 4)), expected)