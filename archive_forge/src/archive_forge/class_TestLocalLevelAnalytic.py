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
class TestLocalLevelAnalytic:

    @classmethod
    def setup_class(cls, **kwargs):
        cls.mod, cls.ssm = model_local_level(**kwargs)
        cls.res = cls.ssm.smooth()

    def test_results(self):
        ssm = self.ssm
        res = self.res
        y1 = ssm.endog[0, 0]
        sigma2_y = ssm['obs_cov', 0, 0]
        sigma2_mu = ssm['state_cov', 0, 0]
        assert_allclose(res.predicted_state_cov[0, 0, 0], 0)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 0], 1)
        assert_allclose(res.forecasts_error[0, 0], y1)
        assert_allclose(res.forecasts_error_cov[0, 0, 0], sigma2_y)
        assert_allclose(res.forecasts_error_diffuse_cov[0, 0, 0], 1)
        assert_allclose(res.kalman_gain[0, 0, 0], 1)
        assert_allclose(res.predicted_state[0, 1], y1)
        assert_allclose(res.predicted_state_cov[0, 0, 1], sigma2_y + sigma2_mu)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], 0)
        assert_equal(res.nobs_diffuse, 1)