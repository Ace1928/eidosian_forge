import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
class MLEGLS(GenericLikelihoodModel):
    """ARMA model with exact loglikelhood for short time series

    Inverts (nobs, nobs) matrix, use only for nobs <= 200 or so.

    This class is a pattern for small sample GLS-like models. Intended use
    for loglikelihood of initial observations for ARMA.



    TODO:
    This might be missing the error variance. Does it assume error is
       distributed N(0,1)
    Maybe extend to mean handling, or assume it is already removed.
    """

    def _params2cov(self, params, nobs):
        """get autocovariance matrix from ARMA regression parameter

        ar parameters are assumed to have rhs parameterization

        """
        ar = np.r_[[1], -params[:self.nar]]
        ma = np.r_[[1], params[-self.nma:]]
        autocov = arma_acovf(ar, ma, nobs=nobs)
        autocov = autocov[:nobs]
        sigma = toeplitz(autocov)
        return sigma

    def loglike(self, params):
        sig = self._params2cov(params[:-1], self.nobs)
        sig = sig * params[-1] ** 2
        loglik = mvn_loglike(self.endog, sig)
        return loglik

    def fit_invertible(self, *args, **kwds):
        res = self.fit(*args, **kwds)
        ma = np.r_[[1], res.params[self.nar:self.nar + self.nma]]
        mainv, wasinvertible = invertibleroots(ma)
        if not wasinvertible:
            start_params = res.params.copy()
            start_params[self.nar:self.nar + self.nma] = mainv[1:]
            res = self.fit(start_params=start_params)
        return res