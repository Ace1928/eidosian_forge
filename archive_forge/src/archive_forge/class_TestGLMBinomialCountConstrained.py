import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class TestGLMBinomialCountConstrained(ConstrainedCompareMixin):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = np.asarray(data.exog)
        data.endog = np.asarray(data.endog)
        exog = add_constant(data.exog, prepend=True)
        offset = np.ones(len(data.endog))
        exog_keep = exog[:, :-5]
        cls.mod2 = GLM(data.endog, exog_keep, family=family.Binomial(), offset=offset)
        cls.mod1 = GLM(data.endog, exog, family=family.Binomial(), offset=offset)
        cls.init()

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        k = cls.mod1.exog.shape[1]
        cls.idx_p_uc = np.arange(k - 5)
        constraints = np.eye(k)[-5:]
        cls.res1 = cls.mod1.fit_constrained(constraints)

    def test_resid(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.resid_response, res2.resid_response, rtol=1e-08)

    def test_glm_attr(self):
        for attr in ['llf', 'null_deviance', 'aic', 'df_resid', 'df_model', 'pearson_chi2', 'scale']:
            assert_allclose(getattr(self.res1, attr), getattr(self.res2, attr), rtol=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            assert_allclose(self.res1.bic, self.res2.bic, rtol=1e-10)

    def test_wald(self):
        res1 = self.res1
        res2 = self.res2
        k1 = len(res1.params)
        k2 = len(res2.params)
        use_f = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValueWarning)
            wt2 = res2.wald_test(np.eye(k2)[1:], use_f=use_f, scalar=True)
            wt1 = res1.wald_test(np.eye(k1)[1:], use_f=use_f, scalar=True)
        assert_allclose(wt2.pvalue, wt1.pvalue, atol=1e-20)
        assert_allclose(wt2.statistic, wt1.statistic, rtol=1e-08)
        assert_equal(wt2.df_denom, wt1.df_denom)
        use_f = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValueWarning)
            wt2 = res2.wald_test(np.eye(k2)[1:], use_f=use_f, scalar=True)
            wt1 = res1.wald_test(np.eye(k1)[1:], use_f=use_f, scalar=True)
        assert_allclose(wt2.pvalue, wt1.pvalue, rtol=1)
        assert_allclose(wt2.statistic, wt1.statistic, rtol=1e-08)
        assert_equal(wt2.df_denom, wt1.df_denom)
        assert_equal(wt2.df_num, wt1.df_num)
        assert_equal(wt2.summary()[-30:], wt1.summary()[-30:])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', ValueWarning)
            warnings.simplefilter('ignore', RuntimeWarning)
            self.res1.summary()
            self.res1.summary2()