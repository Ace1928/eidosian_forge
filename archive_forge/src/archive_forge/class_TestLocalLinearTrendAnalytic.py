from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
class TestLocalLinearTrendAnalytic:

    @classmethod
    def setup_class(cls, **kwargs):
        cls.mod, cls.ssm = model_local_linear_trend(**kwargs)
        cls.res = cls.ssm.smooth()

    def test_results(self):
        ssm = self.ssm
        res = self.res
        y1, y2, y3 = ssm.endog[0, :3]
        sigma2_y = ssm['obs_cov', 0, 0]
        sigma2_mu, sigma2_beta = np.diagonal(ssm['state_cov'])
        assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
        assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))
        q_mu = sigma2_mu / sigma2_y
        q_beta = sigma2_beta / sigma2_y
        assert_allclose(res.forecasts_error[0, 0], y1)
        assert_allclose(res.kalman_gain[:, 0, 0], [1, 0])
        assert_allclose(res.predicted_state[:, 1], [y1, 0])
        P2 = sigma2_y * np.array([[1 + q_mu, 0], [0, q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 1], P2)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], np.ones((2, 2)))
        assert_allclose(res.predicted_state[:, 2], [2 * y2 - y1, y2 - y1])
        P3 = sigma2_y * np.array([[5 + 2 * q_mu + q_beta, 3 + q_mu + q_beta], [3 + q_mu + q_beta, 2 + q_mu + 2 * q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 2], P3)
        assert_allclose(res.predicted_diffuse_state_cov[:, :, 2], np.zeros((2, 2)))
        assert_equal(res.nobs_diffuse, 2)