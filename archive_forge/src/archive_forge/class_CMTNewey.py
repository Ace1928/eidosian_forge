import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
class CMTNewey:
    """generic moment test for GMM

    This is a class to calculate and hold the various results

    This is based on Newey 1985 on GMM.
    Lemma 1:
    Theorem 1

    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and methods will likely be changed

    Parameters
    ----------
    moments : ndarray, 1-D
        moments that are tested to be zero. They do not need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of all moments. This
        can be a heteroscedasticity or correlation robust covariance estimate,
        i.e. the inner part of a sandwich covariance.
    weights : ndarray
        Weights used in the GMM estimation.
    transf_mt : ndarray
        This defines the test moments where `transf_mt` is the matrix that
        defines a Linear combination of moments that have expected value equal
        to zero under the Null hypothesis.

    Notes
    -----
    The one letter names in Newey 1985 are

    moments, g :
    cov_moments, V :
    moments_deriv, H :
    weights, W :
    transf_mt, L :
        linear transformation to get the test condition from the moments

    not used, add as argument to methods or __init__?
    K cov for misspecification
    or mispecification_deriv

    This follows the GMM version in Newey 1985a, not the MLE version in
    Newey 1985b. Newey uses the generalized information matrix equality in the
    MLE version Newey (1985b).

    Newey 1985b Lemma 1 does not impose correctly specified likelihood, but
    assumes it in the following. Lemma 1 in both articles are essentially the
    same assuming D = H' W.

    References
    ----------
    - Newey 1985a, Generalized Method of Moment specification testing,
      Journal of Econometrics
    - Newey 1985b, Maximum Likelihood Specification Testing and Conditional
      Moment Tests, Econometrica
    """

    def __init__(self, moments, cov_moments, moments_deriv, weights, transf_mt):
        self.moments = moments
        self.cov_moments = cov_moments
        self.moments_deriv = moments_deriv
        self.weights = weights
        self.transf_mt = transf_mt
        self.moments_constraint = transf_mt.dot(moments)
        self.htw = moments_deriv.T.dot(weights)
        self.k_moments = self.moments.shape[-1]
        self.k_constraints = self.transf_mt.shape[0]

    @cache_readonly
    def asy_transf_params(self):
        moments_deriv = self.moments_deriv
        htw = self.htw
        res = np.linalg.solve(htw.dot(moments_deriv), htw)
        return -res

    @cache_readonly
    def project_w(self):
        moments_deriv = self.moments_deriv
        res = moments_deriv.dot(self.asy_transf_params)
        res += np.eye(res.shape[0])
        return res

    @cache_readonly
    def asy_transform_mom_constraints(self):
        res = self.transf_mt.dot(self.project_w)
        return res

    @cache_readonly
    def asy_cov_moments(self):
        """

        `sqrt(T) * g_T(b_0) asy N(K delta, V)`

        mean is not implemented,
        V is the same as cov_moments in __init__ argument
        """
        return self.cov_moments

    @cache_readonly
    def cov_mom_constraints(self):
        transf = self.asy_transform_mom_constraints
        return transf.dot(self.asy_cov_moments).dot(transf.T)

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
        diff = self.moments_constraint
        cov = self.cov_mom_constraints
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        df = self.rank_cov_mom_constraints
        pval = stats.chi2.sf(stat, df)
        return (stat, pval, df)