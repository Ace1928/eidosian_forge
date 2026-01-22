import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import (
from statsmodels.discrete.discrete_model import (
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy
class TruncatedLFGeneric(CountModel):
    __doc__ = '\n    Generic Truncated model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, truncation=0, offset=None, exposure=None, missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        mask = self.endog > truncation
        self.exog = self.exog[mask]
        self.endog = self.endog[mask]
        if offset is not None:
            self.offset = self.offset[mask]
        if exposure is not None:
            self.exposure = self.exposure[mask]
        self.trunc = truncation
        self.truncation = truncation
        self._init_keys.extend(['truncation'])
        self._null_drop_keys = []

    def loglike(self, params):
        """
        Loglikelihood of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----

        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----

        """
        llf_main = self.model_main.loglikeobs(params)
        yt = self.trunc + 1
        pmf = self.predict(params, which='prob-base', y_values=np.arange(yt)).sum(-1)
        log_1_m_pmf = np.full_like(pmf, -np.inf)
        loc = pmf > 1
        log_1_m_pmf[loc] = np.nan
        loc = pmf < 1
        log_1_m_pmf[loc] = np.log(1 - pmf[loc])
        llf = llf_main - log_1_m_pmf
        return llf

    def score_obs(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)
        pmf = np.zeros_like(self.endog, dtype=np.float64)
        score_trunc = np.zeros_like(score_main, dtype=np.float64)
        for i in range(self.trunc + 1):
            model = self.model_main.__class__(np.ones_like(self.endog) * i, self.exog, offset=getattr(self, 'offset', None), exposure=getattr(self, 'exposure', None))
            pmf_i = np.exp(model.loglikeobs(params))
            score_trunc += (model.score_obs(params).T * pmf_i).T
            pmf += pmf_i
        dparams = score_main + (score_trunc.T / (1 - pmf)).T
        return dparams

    def score(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=ConvergenceWarning)
                start_params = model.fit(disp=0).params
        k_params = self.df_model + 1 + self.k_extra
        self.df_resid = self.endog.shape[0] - k_params
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, disp=disp, full_output=full_output, callback=lambda x: x, **kwargs)
        zipfit = self.result_class(self, mlefit._results)
        result = self.result_class_wrapper(zipfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result
    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)
        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = self.result_class_reg(self, cntfit)
        else:
            raise TypeError('argument method == %s, which is not handled' % method)
        return self.result_class_reg_wrapper(discretefit)
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Truncated model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        return approx_hess(params, self.loglike)

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        """
        Predict response variable or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.
        which : str (optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob-trunc' : probability of truncation. This is the probability
              of observing a zero count implied
              by the truncation model.
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).
              The probabilities in the truncated region are zero.
            - 'prob-base' : probabilities for untruncated base distribution.
              The probabilities are for each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).


        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        predicted values

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        exog, offset, exposure = self._get_predict_arrays(exog=exog, offset=offset, exposure=exposure)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            mu = np.exp(linpred)
            if self.truncation == 0:
                prob_main = self.model_main._prob_nonzero(mu, params)
                return mu / prob_main
            elif self.truncation == -1:
                return mu
            elif self.truncation > 0:
                counts = np.atleast_2d(np.arange(0, self.truncation + 1))
                probs = self.model_main.predict(params, exog=exog, exposure=np.exp(exposure), offset=offset, which='prob', y_values=counts)
                prob_tregion = probs.sum(1)
                mean_tregion = (np.arange(self.truncation + 1) * probs).sum(1)
                mean = (mu - mean_tregion) / (1 - prob_tregion)
                return mean
            else:
                raise ValueError('unsupported self.truncation')
        elif which == 'linear':
            return linpred
        elif which == 'mean-main':
            return np.exp(linpred)
        elif which == 'prob':
            if y_values is not None:
                counts = np.atleast_2d(y_values)
            else:
                counts = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = np.exp(linpred)[:, None]
            if self.k_extra == 0:
                probs = self.model_dist.pmf(counts, mu, self.trunc)
            elif self.k_extra == 1:
                p = self.model_main.parameterization
                probs = self.model_dist.pmf(counts, mu, params[-1], p, self.trunc)
            else:
                raise ValueError('k_extra is not 0 or 1')
            return probs
        elif which == 'prob-base':
            if y_values is not None:
                counts = np.asarray(y_values)
            else:
                counts = np.arange(0, np.max(self.endog) + 1)
            probs = self.model_main.predict(params, exog=exog, exposure=np.exp(exposure), offset=offset, which='prob', y_values=counts)
            return probs
        elif which == 'var':
            mu = np.exp(linpred)
            counts = np.atleast_2d(np.arange(0, self.truncation + 1))
            probs = self.model_main.predict(params, exog=exog, exposure=np.exp(exposure), offset=offset, which='prob', y_values=counts)
            prob_tregion = probs.sum(1)
            mean_tregion = (np.arange(self.truncation + 1) * probs).sum(1)
            mean = (mu - mean_tregion) / (1 - prob_tregion)
            mnc2_tregion = (np.arange(self.truncation + 1) ** 2 * probs).sum(1)
            vm = self.model_main._var(mu, params)
            mnc2 = (mu ** 2 + vm - mnc2_tregion) / (1 - prob_tregion)
            v = mnc2 - mean ** 2
            return v
        else:
            raise ValueError('argument which == %s not handled' % which)