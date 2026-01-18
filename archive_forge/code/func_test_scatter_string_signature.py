import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_scatter_string_signature(self):
    scatter = Scatter([], 'a', 'b')
    self.assertEqual(scatter.kdims, [Dimension('a')])
    self.assertEqual(scatter.vdims, [Dimension('b')])