from unittest import skipIf
import pandas as pd
import numpy as np
from holoviews import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std
def test_rolling_outliers_std_dates(self):
    outliers = rolling_outlier_std(self.date_outliers, rolling_window=2, sigma=1)
    self.assertEqual(outliers, Scatter([(pd.Timestamp('2016-01-05'), 10)]))