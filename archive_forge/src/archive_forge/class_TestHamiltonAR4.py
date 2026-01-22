import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
class TestHamiltonAR4(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.754673, 0.095915, -0.358811, 1.163516, np.exp(-0.262658) ** 2, 0.013486, -0.057521, -0.246983, -0.212923], 'llf': -181.26339, 'llf_fit': -181.26339, 'llf_fit_em': -183.85444, 'bse_oim': np.r_[0.0965189, 0.0377362, 0.2645396, 0.0745187, np.nan, 0.1199942, 0.137663, 0.1069103, 0.1105311]}
        super().setup_class(true, rgnp, k_regimes=2, order=4, switching_ar=False)

    def test_filtered_regimes(self):
        res = self.result
        assert_equal(len(res.filtered_marginal_probabilities[:, 1]), self.model.nobs)
        assert_allclose(res.filtered_marginal_probabilities[:, 1], hamilton_ar4_filtered, atol=1e-05)

    def test_smoothed_regimes(self):
        res = self.result
        assert_equal(len(res.smoothed_marginal_probabilities[:, 1]), self.model.nobs)
        assert_allclose(res.smoothed_marginal_probabilities[:, 1], hamilton_ar4_smoothed, atol=1e-05)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:4], self.true['bse_oim'][:4], atol=1e-06)
        assert_allclose(bse[6:], self.true['bse_oim'][6:], atol=1e-06)