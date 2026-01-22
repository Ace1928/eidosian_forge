import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
class TestSUR(CheckDynamicFactor):
    """
    Test for a seemingly unrelated regression model (i.e. no factors) with
    errors cross-sectionally, but not auto-, correlated
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sur.copy()
        true['predict'] = output_results.iloc[1:][['predict_sur_1', 'predict_sur_2', 'predict_sur_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sur_1', 'dyn_predict_sur_2', 'dyn_predict_sur_3']]
        exog = np.c_[np.ones((75, 1)), (np.arange(75) + 2)[:, np.newaxis]]
        super().setup_class(true, k_factors=0, factor_order=0, exog=exog, error_cov_type='unstructured')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse[:6], self.true['var_oim'][:6], atol=1e-05)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_dynamic_predict(exog=exog)