import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
class CMTTauchen:
    """generic moment tests or conditional moment tests for Quasi-MLE

    This is a generic class based on Tauchen 1985

    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and of methods will likely be changed

    Parameters
    ----------
    score : ndarray, 1-D
        moment condition used in estimation, score of log-likelihood function
    score_deriv : ndarray
        derivative of score function with respect to the parameters that are
        estimated. This is the Hessian in quasi-maximum likelihood
    moments : ndarray, 1-D
        moments that are tested to be zero. They do not need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of score and test
        moments. This can be a heteroscedasticity or correlation robust
        covariance estimate, i.e. the inner part of a sandwich covariance.
    """

    def __init__(self, score, score_deriv, moments, moments_deriv, cov_moments):
        self.score = score
        self.score_deriv = score_deriv
        self.moments = moments
        self.moments_deriv = moments_deriv
        self.cov_moments_all = cov_moments
        self.k_moments_test = moments.shape[-1]
        self.k_params = score.shape[-1]
        self.k_moments_all = self.k_params + self.k_moments_test

    @cache_readonly
    def cov_params_all(self):
        m_deriv = np.zeros((self.k_moments_all, self.k_moments_all))
        m_deriv[:self.k_params, :self.k_params] = self.score_deriv
        m_deriv[self.k_params:, :self.k_params] = self.moments_deriv
        m_deriv[self.k_params:, self.k_params:] = np.eye(self.k_moments_test)
        m_deriv_inv = np.linalg.inv(m_deriv)
        cov = m_deriv_inv.dot(self.cov_moments_all.dot(m_deriv_inv.T))
        return cov

    @cache_readonly
    def cov_mom_constraints(self):
        return self.cov_params_all[self.k_params:, self.k_params:]

    @cache_readonly
    def rank_cov_mom_constraints(self):
        return np.linalg.matrix_rank(self.cov_mom_constraints)

    def ztest(self):
        """statistic, p-value and degrees of freedom of separate moment test

        currently two sided test only

        TODO: This can use generic ztest/ttest features and return
        ContrastResults
        """
        diff = self.moments_constraint
        bse = np.sqrt(np.diag(self.cov_mom_constraints))
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat)) * 2
        return (stat, pval)

    @cache_readonly
    def chisquare(self):
        """statistic, p-value and degrees of freedom of joint moment test
        """
        diff = self.moments
        cov = self.cov_mom_constraints
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        df = self.rank_cov_mom_constraints
        pval = stats.chi2.sf(stat, df)
        return (stat, pval, df)