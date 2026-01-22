from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class DistQuantilesGMM(GMM):
    """
    Estimate distribution parameters by GMM based on matching quantiles

    Currently mainly to try out different requirements for GMM when we cannot
    calculate the optimal weighting matrix.

    """

    def __init__(self, endog, exog, instrument, **kwds):
        super().__init__(endog, exog, instrument)
        self.epsilon_iter = 1e-05
        self.distfn = kwds['distfn']
        self.endog = endog
        if 'pquant' not in kwds:
            self.pquant = pquant = np.array([0.01, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 0.99])
        else:
            self.pquant = pquant = kwds['pquant']
        self.xquant = np.array([stats.scoreatpercentile(endog, p) for p in pquant * 100])
        self.nmoms = len(self.pquant)
        self.endog = endog
        self.exog = exog
        self.instrument = instrument
        self.results = GMMResults(model=self)
        self.epsilon_iter = 1e-06

    def fitstart(self):
        distfn = self.distfn
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(self.endog)
        else:
            start = [1] * distfn.numargs + [0.0, 1.0]
        return np.asarray(start)

    def momcond(self, params):
        """moment conditions for estimating distribution parameters by matching
        quantiles, defines as many moment conditions as quantiles.

        Returns
        -------
        difference : ndarray
            difference between theoretical and empirical quantiles

        Notes
        -----
        This can be used for method of moments or for generalized method of
        moments.

        """
        if len(params) == 2:
            loc, scale = params
        elif len(params) == 3:
            shape, loc, scale = params
        else:
            pass
        pq, xq = (self.pquant, self.xquant)
        cdfdiff = self.distfn.cdf(xq, *params) - pq
        return np.atleast_2d(cdfdiff)

    def fitonce(self, start=None, weights=None, has_optimal_weights=False):
        """fit without estimating an optimal weighting matrix and return results

        This is a convenience function that calls fitgmm and covparams with
        a given weight matrix or the identity weight matrix.
        This is useful if the optimal weight matrix is know (or is analytically
        given) or if an optimal weight matrix cannot be calculated.

        (Developer Notes: this function could go into GMM, but is needed in this
        class, at least at the moment.)

        Parameters
        ----------


        Returns
        -------
        results : GMMResult instance
            result instance with params and _cov_params attached

        See Also
        --------
        fitgmm
        cov_params

        """
        if weights is None:
            weights = np.eye(self.nmoms)
        params = self.fitgmm(start=start)
        self.results.params = params
        self.results.wargs = {}
        self.results.options_other = {'weights_method': 'cov'}
        _cov_params = self.results.cov_params(weights=weights, has_optimal_weights=has_optimal_weights)
        self.results.weights = weights
        self.results.jval = self.gmmobjective(params, weights)
        self.results.options_other.update({'has_optimal_weights': has_optimal_weights})
        return self.results