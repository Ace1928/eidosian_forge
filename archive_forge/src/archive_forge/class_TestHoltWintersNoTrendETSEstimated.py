import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersNoTrendETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4, concentrate_scale=False)
        params = np.r_[results_params['aust_ets3']['alpha'], results_params['aust_ets3']['gamma'], results_params['aust_ets3']['sigma2'], results_params['aust_ets3']['l0'], results_params['aust_ets3']['s0_0'], results_params['aust_ets3']['s0_1'], results_params['aust_ets3']['s0_2']]
        res = mod.filter(params)
        super().setup_class('aust_ets3', res)

    def test_conf_int(self):
        j = np.arange(1, 5)
        alpha, gamma, sigma2 = self.res.params[:3]
        c = np.r_[0, alpha + gamma * (j % 4 == 0).astype(int)]
        se = (sigma2 * (1 + np.cumsum(c ** 2))) ** 0.5
        assert_allclose(self.forecast.se_mean, se)

    def test_mle_estimates(self):
        start_params = [0.5, 0.4, 4, 32, 2.3, -2, -9]
        mle_res = self.res.model.fit(start_params, disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)