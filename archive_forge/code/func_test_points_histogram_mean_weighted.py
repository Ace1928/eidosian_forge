import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_points_histogram_mean_weighted(self):
    points = Points([float(i) for i in range(10)])
    op_hist = histogram(points, num_bins=3, weight_dimension='y', mean_weighted=True, normed=True)
    hist = Histogram(([1.0, 4.0, 7.5], [0, 3, 6, 9]), vdims=['y'])
    self.assertEqual(op_hist, hist)