import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
class ShortPanelGLS(GLS):
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group, sigma_i=None):
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups
        nobs_i = len(endog) / self.n_groups
        if sigma_i is None:
            sigma_i = np.eye(int(nobs_i))
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        super(self.__class__, self).__init__(endog, exog, sigma=None)

    def get_within_cov(self, resid):
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups

    def whiten_groups(self, x, cholsigmainv_i):
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def _fit_ols(self):
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled

    def _fit_old(self):
        res_pooled = self._fit_ols()
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1

    def whiten(self, x):
        wx = whiten_individuals_loop(x, self.cholsigmainv_i, self.group.group_iter)
        return wx

    def fit_iterative(self, maxiter=3):
        """
        Perform an iterative two-step procedure to estimate the GLS model.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) once does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
        if maxiter < 1:
            raise ValueError('maxiter needs to be at least 1')
        import collections
        self.history = collections.defaultdict(list)
        for i in range(maxiter):
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            results = self.fit()
            self.history['self_params'].append(results.params)
            if not i == maxiter - 1:
                self.results_old = results
                sigma_i = self.get_within_cov(results.resid)
                self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
                self.initialize()
        return results