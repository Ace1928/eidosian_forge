import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_select_tuple(self):
    ds = self.element((self.grid_xs, self.grid_ys[:2], self.grid_zs[:2]), ['x', 'y'], ['z'])
    self.assertEqual(self.dataset_grid.select(y=(0, 0.25)), ds)