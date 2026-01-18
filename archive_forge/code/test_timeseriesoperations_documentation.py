from unittest import skipIf
import pandas as pd
import numpy as np
from holoviews import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std

    Tests for the various timeseries operations including rolling,
    resample and rolling_outliers_std.
    