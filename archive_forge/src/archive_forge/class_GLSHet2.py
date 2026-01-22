import numpy as np
from statsmodels.regression.linear_model import OLS, GLS, WLS
class GLSHet2(GLS):
    """WLS with heteroscedasticity that depends on explanatory variables

    note: mixing GLS sigma and weights for heteroscedasticity might not make
    sense

    I think rewriting following the pattern of GLSAR is better
    stopping criteria: improve in GLSAR also, e.g. change in rho

    """

    def __init__(self, endog, exog, exog_var, sigma=None):
        self.exog_var = atleast_2dcols(exog_var)
        super(self.__class__, self).__init__(endog, exog, sigma=sigma)

    def fit(self, lambd=1.0):
        res_gls = GLS(self.endog, self.exog, sigma=self.sigma).fit()
        res_resid = OLS(res_gls.resid ** 2, self.exog_var).fit()
        res_wls = WLS(self.endog, self.exog, weights=1.0 / res_resid.fittedvalues).fit()
        res_wls._results.results_residual_regression = res_resid
        return res_wls