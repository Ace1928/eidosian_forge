import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
class TestTwoPeakLLHNoExog:

    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        pdf_a = stats.halfcauchy(loc=0, scale=1)
        pdf_b = stats.uniform(loc=0, scale=100)
        n_a = 50
        n_b = 200
        params = [n_a, n_b]
        X = np.concatenate([pdf_a.rvs(size=n_a), pdf_b.rvs(size=n_b)])[:, np.newaxis]
        cls.X = X
        cls.params = params
        cls.pdf_a = pdf_a
        cls.pdf_b = pdf_b

    def test_fit(self):
        np.random.seed(42)
        llh_noexog = TwoPeakLLHNoExog(self.X, signal=self.pdf_a, background=self.pdf_b)
        res = llh_noexog.fit()
        assert_allclose(res.params, self.params, rtol=0.1)
        assert res.df_resid == 248
        assert res.df_model == 0
        res_bs = res.bootstrap(nrep=50)
        assert_allclose(res_bs[2].mean(0), self.params, rtol=0.1)
        res.summary()