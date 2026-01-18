from unittest import skipIf
import pandas as pd
import numpy as np
from holoviews import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std
@scipy_skip
def test_roll_date_with_window_type(self):
    rolled = rolling(self.date_curve, rolling_window=3, window_type='triang')
    rolled_vals = [np.nan, 2, 3, 4, 5, 6, np.nan]
    self.assertEqual(rolled, Curve((self.dates, rolled_vals)))