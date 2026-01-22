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
class TestFriedmanStateRegression(Friedman):
    """
    Notes
    -----

    MLE is not very close and standard errors are not very close for any set of
    parameters.

    This is likely because we're comparing against the model where the
    regression coefficients are also estimated by MLE. So this test should be
    considered just a very basic "sanity" test.
    """

    @classmethod
    def setup_class(cls):
        true = dict(results_sarimax.friedman2_mle)
        exog = add_constant(true['data']['m2']) / 10.0
        true['mle_params_exog'] = true['params_exog'][:]
        true['mle_se_exog'] = true['se_exog_opg'][:]
        true['params_exog'] = []
        true['se_exog'] = []
        super().setup_class(true, exog=exog, mle_regression=False)
        cls.true_params = np.r_[true['params_exog'], true['params_ar'], true['params_ma'], true['params_variance']]
        cls.result = cls.model.filter(cls.true_params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(result.params, self.result.params, atol=0.1, rtol=0.2)

    def test_regression_parameters(self):
        assert_almost_equal(self.result.filter_results.filtered_state[-2:, -1] / 10.0, self.true['mle_params_exog'], 1)

    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse(self):
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        assert_allclose(self.result.bse[0], self.true['se_ar_opg'], atol=0.01)
        assert_allclose(self.result.bse[1], self.true['se_ma_opg'], atol=0.01)

    def test_bse_approx(self):
        bse = self.result._cov_params_approx(approx_complex_step=True).diagonal() ** 0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=0.1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=0.1)

    def test_bse_oim(self):
        bse = self.result._cov_params_oim().diagonal() ** 0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=0.1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=0.1)