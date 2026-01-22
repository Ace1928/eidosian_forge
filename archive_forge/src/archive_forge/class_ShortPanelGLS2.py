import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
class ShortPanelGLS2:
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group):
        self.endog = endog
        self.exog = exog
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups

    def fit_ols(self):
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled

    def get_within_cov(self, resid):
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups

    def whiten_groups(self, x, cholsigmainv_i):
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def fit(self):
        res_pooled = self.fit_ols()
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1