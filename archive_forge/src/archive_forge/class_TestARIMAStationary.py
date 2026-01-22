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
class TestARIMAStationary(ARIMA):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(results_sarimax.wpi1_stationary)

    def test_bse(self):
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-07)
        assert_allclose(self.result.bse[2], self.true['se_ma_opg'], atol=1e-07)

    def test_bse_approx(self):
        bse = self.result._cov_params_approx(approx_complex_step=True).diagonal() ** 0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-07)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-07)

    def test_bse_oim(self):
        oim_bse = self.result.cov_params_oim.diagonal() ** 0.5
        assert_allclose(oim_bse[1], self.true['se_ar_oim'], atol=0.001)
        assert_allclose(oim_bse[2], self.true['se_ma_oim'], atol=0.01)

    def test_bse_robust(self):
        robust_oim_bse = self.result.cov_params_robust_oim.diagonal() ** 0.5
        cpra = self.result.cov_params_robust_approx
        robust_approx_bse = cpra.diagonal() ** 0.5
        true_robust_bse = np.r_[self.true['se_ar_robust'], self.true['se_ma_robust']]
        assert_allclose(robust_oim_bse[1:3], true_robust_bse, atol=0.01)
        assert_allclose(robust_approx_bse[1:3], true_robust_bse, atol=0.001)