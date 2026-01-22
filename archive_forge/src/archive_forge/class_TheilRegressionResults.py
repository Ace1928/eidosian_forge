from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
class TheilRegressionResults(RegressionResults):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.df_model = self.hatmatrix_trace() - 1
        self.df_resid = self.model.endog.shape[0] - self.df_model - 1

    @cache_readonly
    def hatmatrix_diag(self):
        """diagonal of hat matrix

        diag(X' xpxi X)

        where xpxi = (X'X + sigma2_e * lambd * sigma_prior)^{-1}

        Notes
        -----

        uses wexog, so this includes weights or sigma - check this case

        not clear whether I need to multiply by sigmahalf, i.e.

        (W^{-0.5} X) (X' W X)^{-1} (W^{-0.5} X)'  or
        (W X) (X' W X)^{-1} (W X)'

        projection y_hat = H y    or in terms of transformed variables (W^{-0.5} y)

        might be wrong for WLS and GLS case
        """
        xpxi = self.model.normalized_cov_params
        return (self.model.wexog * np.dot(xpxi, self.model.wexog.T).T).sum(1)

    def hatmatrix_trace(self):
        """trace of hat matrix
        """
        return self.hatmatrix_diag.sum()

    @cache_readonly
    def gcv(self):
        return self.mse_resid / (1.0 - self.hatmatrix_trace() / self.nobs) ** 2

    @cache_readonly
    def cv(self):
        return ((self.resid / (1.0 - self.hatmatrix_diag)) ** 2).sum() / self.nobs

    @cache_readonly
    def aicc(self):
        aic = np.log(self.mse_resid) + 1
        eff_dof = self.nobs - self.hatmatrix_trace() - 2
        if eff_dof > 0:
            adj = 2 * (1.0 + self.hatmatrix_trace()) / eff_dof
        else:
            adj = np.inf
        return aic + adj

    def test_compatibility(self):
        """Hypothesis test for the compatibility of prior mean with data
        """
        res_ols = OLS(self.model.endog, self.model.exog).fit()
        r_mat = self.model.r_matrix
        r_diff = self.model.q_matrix - r_mat.dot(res_ols.params)[:, None]
        ols_cov_r = res_ols.cov_params(r_matrix=r_mat)
        statistic = r_diff.T.dot(np.linalg.solve(ols_cov_r + self.model.sigma_prior, r_diff))
        from scipy import stats
        df = np.linalg.matrix_rank(self.model.sigma_prior)
        pvalue = stats.chi2.sf(statistic, df)
        return (statistic, pvalue, df)

    def share_data(self):
        """a measure for the fraction of the data in the estimation result

        The share of the prior information is `1 - share_data`.

        Returns
        -------
        share : float between 0 and 1
            share of data defined as the ration between effective degrees of
            freedom of the model and the number (TODO should be rank) of the
            explanatory variables.
        """
        return (self.df_model + 1) / self.model.rank