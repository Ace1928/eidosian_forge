import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
class CheckSmoother:

    def test_predict(self):
        assert_almost_equal(self.res_ps.predict(self.x), self.res2.fittedvalues, decimal=13)
        assert_almost_equal(self.res_ps.predict(self.x[:10]), self.res2.fittedvalues[:10], decimal=13)

    def test_coef(self):
        assert_almost_equal(self.res_ps.coef.ravel(), self.res2.params, decimal=14)

    def test_df(self):
        assert_equal(self.res_ps.df_model(), self.res2.df_model + 1)
        assert_equal(self.res_ps.df_fit(), self.res2.df_model + 1)
        assert_equal(self.res_ps.df_resid(), self.res2.df_resid)