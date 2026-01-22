import numpy as np
from statsmodels.datasets.ccard.data import load_pandas
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.tools.tools import add_constant
class TestOaxaca:

    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(endog, exog, 3)

    def test_results(self):
        np.random.seed(0)
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        stata_results_std = np.array([653.10389, 64.584796, 655.0323717])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        endow_var, coef_var, inter_var = self.model.three_fold(True).std
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)
        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)
        np.testing.assert_almost_equal(endow_var, stata_results_std[0], 3)
        np.testing.assert_almost_equal(coef_var, stata_results_std[1], 3)
        np.testing.assert_almost_equal(inter_var, stata_results_std[2], 3)