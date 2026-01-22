import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma
class TArma(Arma):
    """Univariate Arma Model with t-distributed errors

    This inherit all methods except loglike from tsa.arma_mle.Arma

    This uses the standard t-distribution, the implied variance of
    the error is not equal to scale, but ::

        error_variance = df/(df-2)*scale**2

    Notes
    -----
    This might be replaced by a standardized t-distribution with scale**2
    equal to variance

    """

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        """
        Loglikelihood for arma model for each observation, t-distribute

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        errorsest = self.geterrors(params[:-2])
        df = params[-2]
        scale = np.abs(params[-1])
        llike = -stats.t._logpdf(errorsest / scale, df) + np_log(scale)
        return llike

    def fit_mle(self, order, start_params=None, method='nm', maxiter=5000, tol=1e-08, **kwds):
        nar, nma = order
        if start_params is not None:
            if len(start_params) != nar + nma + 2:
                raise ValueError('start_param need sum(order) + 2 elements')
        else:
            start_params = np.concatenate((0.05 * np.ones(nar + nma), [5, 1]))
        res = super().fit_mle(order=order, start_params=start_params, method=method, maxiter=maxiter, tol=tol, **kwds)
        return res