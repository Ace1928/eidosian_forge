import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
class CheckRlmResultsMixin:
    """
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to results.results_rlm

    Covariance matrices were obtained from SAS and are imported from
    results.results_rlm
    """

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)
    decimal_standarderrors = DECIMAL_4

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, self.decimal_standarderrors)

    def test_confidenceintervals(self):
        if not hasattr(self.res2, 'conf_int'):
            pytest.skip('Results from R')
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(), DECIMAL_4)
    decimal_scale = DECIMAL_4

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, self.decimal_scale)

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL_4)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model, self.res2.df_model, DECIMAL_4)
        assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid, DECIMAL_4)

    def test_bcov_unscaled(self):
        if not hasattr(self.res2, 'bcov_unscaled'):
            pytest.skip('No unscaled cov matrix from SAS')
        assert_almost_equal(self.res1.bcov_unscaled, self.res2.bcov_unscaled, DECIMAL_4)
    decimal_bcov_scaled = DECIMAL_4

    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled, self.res2.h1, self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h2, self.res2.h2, self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h3, self.res2.h3, self.decimal_bcov_scaled)

    def test_tvalues(self):
        if not hasattr(self.res2, 'tvalues'):
            pytest.skip('No tvalues in benchmark')
        assert_allclose(self.res1.tvalues, self.res2.tvalues, rtol=0.003)

    def test_tpvalues(self):
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))
        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)