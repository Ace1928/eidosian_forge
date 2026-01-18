import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_points_histogram_bin_range(self):
    points = Points([float(i) for i in range(10)])
    op_hist = histogram(points, num_bins=3, bin_range=(0, 3), normed=True)
    hist = Histogram(([0.25, 0.25, 0.5], [0.0, 1.0, 2.0, 3.0]), vdims=('x_frequency', 'Frequency'))
    self.assertEqual(op_hist, hist)