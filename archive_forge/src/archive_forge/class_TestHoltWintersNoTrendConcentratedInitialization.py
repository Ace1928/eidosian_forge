import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersNoTrendConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series([0.5, 0.49, 32.0, 2.3, -2.1, -9.3], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=0.0001)