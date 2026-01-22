import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltDampedConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True)
        start_params = pd.Series([0.95, 0.0005, 0.9, 15.0, 2.5], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=0.1)