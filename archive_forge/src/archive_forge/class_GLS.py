from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
class GLS(RegressionModel):
    __doc__ = '\n    Generalized Least Squares\n\n    {params}\n    sigma : scalar or array\n        The array or scalar `sigma` is the weighting matrix of the covariance.\n        The default is None for no scaling.  If `sigma` is a scalar, it is\n        assumed that `sigma` is an n x n diagonal matrix with the given\n        scalar, `sigma` as the value of each diagonal element.  If `sigma`\n        is an n-length vector, then `sigma` is assumed to be a diagonal\n        matrix with the given `sigma` on the diagonal.  This should be the\n        same as WLS.\n    {extra_params}\n\n    Attributes\n    ----------\n    pinv_wexog : ndarray\n        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse of `wexog`.\n    cholsimgainv : ndarray\n        The transpose of the Cholesky decomposition of the pseudoinverse.\n    df_model : float\n        p - 1, where p is the number of regressors including the intercept.\n        of freedom.\n    df_resid : float\n        Number of observations n less the number of parameters p.\n    llf : float\n        The value of the likelihood function of the fitted model.\n    nobs : float\n        The number of observations n.\n    normalized_cov_params : ndarray\n        p x p array :math:`(X^{{T}}\\Sigma^{{-1}}X)^{{-1}}`\n    results : RegressionResults instance\n        A property that returns the RegressionResults class if fit.\n    sigma : ndarray\n        `sigma` is the n x n covariance structure of the error terms.\n    wexog : ndarray\n        Design matrix whitened by `cholsigmainv`\n    wendog : ndarray\n        Response variable whitened by `cholsigmainv`\n\n    See Also\n    --------\n    WLS : Fit a linear model using Weighted Least Squares.\n    OLS : Fit a linear model using Ordinary Least Squares.\n\n    Notes\n    -----\n    If sigma is a function of the data making one of the regressors\n    a constant, then the current postestimation statistics will not be correct.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.longley.load()\n    >>> data.exog = sm.add_constant(data.exog)\n    >>> ols_resid = sm.OLS(data.endog, data.exog).fit().resid\n    >>> res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()\n    >>> rho = res_fit.params\n\n    `rho` is a consistent estimator of the correlation of the residuals from\n    an OLS fit of the longley data.  It is assumed that this is the true rho\n    of the AR process data.\n\n    >>> from scipy.linalg import toeplitz\n    >>> order = toeplitz(np.arange(16))\n    >>> sigma = rho**order\n\n    `sigma` is an n x n matrix of the autocorrelation structure of the\n    data.\n\n    >>> gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)\n    >>> gls_results = gls_model.fit()\n    >>> print(gls_results.summary())\n    '.format(params=base._model_params_doc, extra_params=base._missing_param_doc + base._extra_param_doc)

    def __init__(self, endog, exog, sigma=None, missing='none', hasconst=None, **kwargs):
        if type(self) is GLS:
            self._check_kwargs(kwargs)
        sigma, cholsigmainv = _get_sigma(sigma, len(endog))
        super().__init__(endog, exog, missing=missing, hasconst=hasconst, sigma=sigma, cholsigmainv=cholsigmainv, **kwargs)
        self._data_attr.extend(['sigma', 'cholsigmainv'])

    def whiten(self, x):
        """
        GLS whiten method.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        ndarray
            The value np.dot(cholsigmainv,X).

        See Also
        --------
        GLS : Fit a linear model using Generalized Least Squares.
        """
        x = np.asarray(x)
        if self.sigma is None or self.sigma.shape == ():
            return x
        elif self.sigma.ndim == 1:
            if x.ndim == 1:
                return x * self.cholsigmainv
            else:
                return x * self.cholsigmainv[:, None]
        else:
            return np.dot(self.cholsigmainv, x)

    def loglike(self, params):
        """
        Compute the value of the Gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array_like
            The model parameters.

        Returns
        -------
        float
            The value of the log-likelihood function for a GLS Model.

        Notes
        -----
        The log-likelihood function for the normal distribution is

        .. math:: -\\frac{n}{2}\\log\\left(\\left(Y-\\hat{Y}\\right)^{\\prime}
                   \\left(Y-\\hat{Y}\\right)\\right)
                  -\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)
                  -\\frac{1}{2}\\log\\left(\\left|\\Sigma\\right|\\right)

        Y and Y-hat are whitened.
        """
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        if np.any(self.sigma):
            if self.sigma.ndim == 2:
                det = np.linalg.slogdet(self.sigma)
                llf -= 0.5 * det[1]
            else:
                llf -= 0.5 * np.sum(np.log(self.sigma))
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute weights for calculating Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        if self.sigma is None or self.sigma.shape == ():
            return np.ones(self.exog.shape[0])
        elif self.sigma.ndim == 1:
            return self.cholsigmainv
        else:
            return np.diag(self.cholsigmainv)

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method='elastic_net', alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False, **kwargs):
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        if self.sigma is not None:
            if self.sigma.ndim == 2:
                var_obs = np.diag(self.sigma)
            elif self.sigma.ndim == 1:
                var_obs = self.sigma
            else:
                raise ValueError('sigma should be 1-dim or 2-dim')
            alpha = alpha * np.sum(1 / var_obs) / len(self.endog)
        rslt = OLS(self.wendog, self.wexog).fit_regularized(method=method, alpha=alpha, L1_wt=L1_wt, start_params=start_params, profile_scale=profile_scale, refit=refit, **kwargs)
        from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)