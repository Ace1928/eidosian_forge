import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class TestSARIMAXStatsmodels:
    """
    Test ARIMA model using SARIMAX class against statsmodels ARIMA class

    Notes
    -----

    Standard errors are quite good for the OPG case.
    """

    @classmethod
    def setup_class(cls):
        cls.true = results_sarimax.wpi1_stationary
        endog = cls.true['data']
        result_a = Bunch()
        result_a.llf = -135.3513139733829
        result_a.aic = 278.7026279467658
        result_a.bic = 289.9513653682555
        result_a.hqic = 283.27183681851653
        result_a.params = np.array([0.74982449, 0.87421135, -0.41202195])
        result_a.bse = np.array([0.29207409, 0.06377779, 0.12208469])
        cls.result_a = result_a
        cls.model_b = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='c', simple_differencing=True, hamilton_representation=True)
        cls.result_b = cls.model_b.fit(disp=-1)

    def test_loglike(self):
        assert_allclose(self.result_b.llf, self.result_a.llf)

    def test_aic(self):
        assert_allclose(self.result_b.aic, self.result_a.aic)

    def test_bic(self):
        assert_allclose(self.result_b.bic, self.result_a.bic)

    def test_hqic(self):
        assert_allclose(self.result_b.hqic, self.result_a.hqic)

    def test_mle(self):
        params_a = self.result_a.params.copy()
        params_a[0] = (1 - params_a[1]) * params_a[0]
        assert_allclose(self.result_b.params[:-1], params_a, atol=5e-05)

    def test_bse(self):
        cpa = self.result_b._cov_params_approx(approx_complex_step=True)
        bse = cpa.diagonal() ** 0.5
        assert_allclose(bse[1:-1], self.result_a.bse[1:], atol=1e-05)

    def test_t_test(self):
        import statsmodels.tools._testing as smt
        smt.check_ttest_tvalues(self.result_b)
        smt.check_ftest_pvalues(self.result_b)