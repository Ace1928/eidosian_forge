from statsmodels.base.model import LikelihoodModel
class TSMLEModel(LikelihoodModel):
    """
    univariate time series model for estimation with maximum likelihood

    Note: This is not working yet
    """

    def __init__(self, endog, exog=None):
        super().__init__(endog, exog)
        self.nar = 1
        self.nma = 1

    def geterrors(self, params):
        raise NotImplementedError

    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Parameters
        ----------
        params : array_like
            The model parameters

        Notes
        -----
        needs to be overwritten by subclass
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector for Arma model
        """
        jac = ndt.Jacobian(self.loglike, stepMax=0.0001)
        return jac(params)[-1]

    def hessian(self, params):
        """
        Hessian of arma model.  Currently uses numdifftools
        """
        Hfun = ndt.Jacobian(self.score, stepMax=0.0001)
        return Hfun(params)[-1]

    def fit(self, start_params=None, maxiter=5000, method='fmin', tol=1e-08):
        """estimate model by minimizing negative loglikelihood

        does this need to be overwritten ?
        """
        if start_params is None and hasattr(self, '_start_params'):
            start_params = self._start_params
        mlefit = super().fit(start_params=start_params, maxiter=maxiter, method=method, tol=tol)
        return mlefit