import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4, initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        trend = aust[:20].rolling(4).mean().rolling(2).mean().shift(-2).dropna()
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(trend[:nobs])
        if not self.mod.trend:
            desired = desired[:1]
        detrended = aust - trend
        initial_seasonal = np.nanmean(detrended.values.reshape(6, 4), axis=0)
        initial_seasonal = initial_seasonal[::-1]
        desired = np.r_[desired, initial_seasonal - np.mean(initial_seasonal)]
        assert_allclose(self.init_heuristic, desired)