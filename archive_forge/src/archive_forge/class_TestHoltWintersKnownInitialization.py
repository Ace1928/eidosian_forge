import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4)
        start_params = pd.Series([0.0005, 0.0004, 0.5, 33.0, 0.4, 2.5, -2.0, -9.0], index=mod.param_names)
        super().setup_class(mod, start_params)