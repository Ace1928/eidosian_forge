import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_points_histogram_cumulative(self):
    arr = np.arange(4)
    points = Points(arr)
    op_hist = histogram(points, cumulative=True, num_bins=3, normed=False)
    hist = Histogram(([0, 1, 2, 3], [1, 2, 4]), vdims=('x_count', 'Count'))
    self.assertEqual(op_hist, hist)