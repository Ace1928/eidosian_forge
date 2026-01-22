import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.discrete._diagnostics_count as dia
from statsmodels.discrete.diagnostic import PoissonDiagnostic
class TestCountDiagnostic:

    @classmethod
    def setup_class(cls):
        expected_params = [1, 1, 0.5]
        np.random.seed(987123)
        nobs = 500
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 0
        offset = 0
        mu_true = np.exp(exog.dot(expected_params[:-1]) + offset)
        endog_poi = np.random.poisson(mu_true / 5)
        model_poi = Poisson(endog_poi, exog)
        res_poi = model_poi.fit(method='bfgs', maxiter=5000, disp=False)
        cls.exog = exog
        cls.endog = endog_poi
        cls.res = res_poi
        cls.nobs = nobs

    def test_count(self):
        tzi1 = dia.test_poisson_zeroinflation_jh(self.res)
        tzi2 = dia.test_poisson_zeroinflation_broek(self.res)
        assert_allclose(tzi1[:2], (tzi2[0] ** 2, tzi2[1]), rtol=1e-05)
        tzi3 = dia.test_poisson_zeroinflation_jh(self.res, self.exog)
        tzi3_1 = (0.7986359783244388, 0.6707773675031893)
        assert_allclose(tzi3, tzi3_1, rtol=0.0005)
        assert_equal(tzi3.df, 2)

    @pytest.mark.matplotlib
    def test_probs(self, close_figures):
        nobs = self.nobs
        probs = self.res.predict_prob()
        freq = np.bincount(self.endog) / nobs
        tzi = dia.test_chisquare_prob(self.res, probs[:, :2])
        tzi1 = (0.387770845, 0.5334734738)
        assert_allclose(tzi[:2], tzi1, rtol=5e-05)
        dia.plot_probs(freq, probs.mean(0))