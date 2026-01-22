from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class GMMResults(LikelihoodModelResults):
    """just a storage class right now"""
    use_t = False

    def __init__(self, *args, **kwds):
        self.__dict__.update(kwds)
        self.nobs = self.model.nobs
        self.df_resid = np.inf
        self.cov_params_default = self._cov_params()

    @cache_readonly
    def q(self):
        """Objective function at params"""
        return self.model.gmmobjective(self.params, self.weights)

    @cache_readonly
    def jval(self):
        """nobs_moms attached by momcond_mean"""
        return self.q * self.model.nobs_moms

    def _cov_params(self, **kwds):
        if 'wargs' not in kwds:
            kwds['wargs'] = self.wargs
        if 'weights_method' not in kwds:
            kwds['weights_method'] = self.options_other['weights_method']
        if 'has_optimal_weights' not in kwds:
            kwds['has_optimal_weights'] = self.options_other['has_optimal_weights']
        gradmoms = self.model.gradient_momcond(self.params)
        moms = self.model.momcond(self.params)
        covparams = self.calc_cov_params(moms, gradmoms, **kwds)
        return covparams

    def calc_cov_params(self, moms, gradmoms, weights=None, use_weights=False, has_optimal_weights=True, weights_method='cov', wargs=()):
        """calculate covariance of parameter estimates

        not all options tried out yet

        If weights matrix is given, then the formula use to calculate cov_params
        depends on whether has_optimal_weights is true.
        If no weights are given, then the weight matrix is calculated with
        the given method, and has_optimal_weights is assumed to be true.

        (API Note: The latter assumption could be changed if we allow for
        has_optimal_weights=None.)

        """
        nobs = moms.shape[0]
        if weights is None:
            weights = self.weights
        else:
            pass
        if use_weights:
            omegahat = weights
        else:
            omegahat = self.model.calc_weightmatrix(moms, weights_method=weights_method, wargs=wargs, params=self.params)
        if has_optimal_weights:
            cov = np.linalg.inv(np.dot(gradmoms.T, np.dot(np.linalg.inv(omegahat), gradmoms)))
        else:
            gw = np.dot(gradmoms.T, weights)
            gwginv = np.linalg.inv(np.dot(gw, gradmoms))
            cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)
        return cov / nobs

    @property
    def bse_(self):
        """standard error of the parameter estimates
        """
        return self.get_bse()

    def get_bse(self, **kwds):
        """standard error of the parameter estimates with options

        Parameters
        ----------
        kwds : optional keywords
            options for calculating cov_params

        Returns
        -------
        bse : ndarray
            estimated standard error of parameter estimates

        """
        return np.sqrt(np.diag(self.cov_params(**kwds)))

    def jtest(self):
        """overidentification test

        I guess this is missing a division by nobs,
        what's the normalization in jval ?
        """
        jstat = self.jval
        nparams = self.params.size
        df = self.model.nmoms - nparams
        return (jstat, stats.chi2.sf(jstat, df), df)

    def compare_j(self, other):
        """overidentification test for comparing two nested gmm estimates

        This assumes that some moment restrictions have been dropped in one
        of the GMM estimates relative to the other.

        Not tested yet

        We are comparing two separately estimated models, that use different
        weighting matrices. It is not guaranteed that the resulting
        difference is positive.

        TODO: Check in which cases Stata programs use the same weigths

        """
        jstat1 = self.jval
        k_moms1 = self.model.nmoms
        jstat2 = other.jval
        k_moms2 = other.model.nmoms
        jdiff = jstat1 - jstat2
        df = k_moms1 - k_moms2
        if df < 0:
            df = -df
            jdiff = -jdiff
        return (jdiff, stats.chi2.sf(jdiff, df), df)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Default is `var_##` for ## in p the number of regressors
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """
        jvalue, jpvalue, jdf = self.jtest()
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['GMM']), ('Date:', None), ('Time:', None), ('No. Observations:', None)]
        top_right = [('Hansen J:', ['%#8.4g' % jvalue]), ('Prob (Hansen J):', ['%#6.3g' % jpvalue])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        return smry