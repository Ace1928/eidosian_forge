import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_points_histogram_explicit_bins(self):
    points = Points([float(i) for i in range(10)])
    op_hist = histogram(points, bins=[0, 1, 3], normed=False)
    hist = Histogram(([0, 1, 3], [1, 3]), vdims=('x_count', 'Count'))
    self.assertEqual(op_hist, hist)