import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_raster_string_signature(self):
    raster = Raster(np.array([[0]]), ['a', 'b'], 'c')
    self.assertEqual(raster.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(raster.vdims, [Dimension('c')])