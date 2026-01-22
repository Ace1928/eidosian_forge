import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._delta_method import NonlinearDeltaCov
class TestDeltacovOLS:

    @classmethod
    def setup_class(cls):
        nobs, k_vars = (100, 4)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        y = x[:, :-1].sum(1) + np.random.randn(nobs)
        cls.res = OLS(y, x).fit()

    def test_method(self):
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params) ** 2
        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        nlm = res._get_wald_nonlinear(fun)
        assert_allclose(nlm.se_vectorized(), nlm.se_vectorized(), rtol=1e-12)
        assert_allclose(nlm.predicted(), nlm.predicted(), rtol=1e-12)
        df = res.df_resid
        t1 = nl.summary(use_t=True, df=df)
        t2 = nlm.summary(use_t=True, df=df)
        assert_equal(str(t1), str(t2))

    def test_ttest(self):
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params)
        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        predicted = nl.predicted()
        se = nl.se_vectorized()
        assert_allclose(predicted, fun(res.params), rtol=1e-12)
        assert_allclose(se, np.sqrt(np.diag(nl.cov())), rtol=1e-12)
        tt = res.t_test(x, use_t=False)
        assert_allclose(predicted, tt.effect, rtol=1e-12)
        assert_allclose(se, tt.sd, rtol=1e-12)
        assert_allclose(nl.conf_int(), tt.conf_int(), rtol=1e-12)
        t1 = nl.summary()
        t2 = tt.summary()
        assert_equal(str(t1), str(t2))
        predicted = nl.predicted()
        se = nl.se_vectorized()
        df = res.df_resid
        tt = res.t_test(x, use_t=True)
        assert_allclose(nl.conf_int(use_t=True, df=df), tt.conf_int(), rtol=1e-12, atol=1e-10)
        t1 = nl.summary(use_t=True, df=df)
        t2 = tt.summary()
        assert_equal(str(t1), str(t2))

    def test_diff(self):
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params) - np.dot(x[:, 1:], params[1:])
        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        assert_allclose(nl.predicted(), res.params[0], rtol=1e-12)
        assert_allclose(nl.se_vectorized(), res.bse[0], rtol=1e-12)