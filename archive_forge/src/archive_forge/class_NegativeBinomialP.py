from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class NegativeBinomialP(CountModel):
    __doc__ = '\n    Generalized Negative Binomial (NB-P) Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    p : scalar\n        P denotes parameterizations for NB-P regression. p=1 for NB-1 and\n        p=2 for NB-2. Default is p=1.\n    ' % {'params': base._model_params_doc, 'extra_params': 'p : scalar\n        P denotes parameterizations for NB regression. p=1 for NB-1 and\n        p=2 for NB-2. Default is p=2.\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n        ' + base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, p=2, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, check_rank=check_rank, **kwargs)
        self.parameterization = p
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        kwds['p'] = self.parameterization
        return kwds

    def _get_exogs(self):
        return (self.exog, None)

    def loglike(self, params):
        """
        Loglikelihood of Generalized Negative Binomial (NB-P) model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generalized Negative Binomial (NB-P) model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** (2 - p)
        a1 = mu_p / alpha
        a2 = mu + a1
        llf = gammaln(y + a1) - gammaln(y + 1) - gammaln(a1) + a1 * np.log(a1) + y * np.log(mu) - (y + a1) * np.log(a2)
        return llf

    def score_obs(self, params):
        """
        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood for each observations.

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        dgpart = digamma(a3) - digamma(a1)
        dgterm = dgpart + np.log(a1 / a2) + 1 - a3 / a2
        dparams = a4 * dgterm - a3 / a2 + y / mu
        dparams = (self.exog.T * mu * dparams).T
        dalpha = -a1 / alpha * dgterm
        return np.concatenate((dparams, np.atleast_2d(dalpha).T), axis=1)

    def score(self, params):
        """
        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def score_factor(self, params, endog=None):
        """
        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood for each observations.

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
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog if endog is None else endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        dgpart = digamma(a3) - digamma(a1)
        dparams = a4 * dgpart - a3 / a2 + y / mu + a4 * (1 - a3 / a2 + np.log(a1 / a2))
        dparams = (mu * dparams).T
        dalpha = -a1 / alpha * (dgpart + np.log(a1 / a2) + 1 - a3 / a2)
        return (dparams, dalpha)

    def hessian(self, params):
        """
        Generalized Negative Binomial (NB-P) model hessian maxtrix of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hessian : ndarray, 2-D
            The hessian matrix of the model.
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        exog = self.exog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        prob = a1 / a2
        lprob = np.log(prob)
        dgpart = digamma(a3) - digamma(a1)
        pgpart = polygamma(1, a3) - polygamma(1, a1)
        dim = exog.shape[1]
        hess_arr = np.zeros((dim + 1, dim + 1))
        coeff = mu ** 2 * ((1 + a4) ** 2 * a3 / a2 ** 2 - a3 / a2 * (p - 1) * a4 / mu - y / mu ** 2 - 2 * a4 * (1 + a4) / a2 + p * a4 / mu * (lprob + dgpart + 2) - a4 / mu * (lprob + dgpart + 1) + a4 ** 2 * pgpart + (-(1 + a4) * a3 / a2 + y / mu + a4 * (lprob + dgpart + 1)) / mu)
        for i in range(dim):
            hess_arr[i, :-1] = np.sum(self.exog[:, :].T * self.exog[:, i] * coeff, axis=1)
        hess_arr[-1, :-1] = (self.exog[:, :].T * mu * a1 * ((1 + a4) * (1 - a3 / a2) / a2 - p * (lprob + dgpart + 2) / mu + p / mu * (a3 + p * a1) / a2 - a4 * pgpart) / alpha).sum(axis=1)
        da2 = a1 * (2 * lprob + 2 * dgpart + 3 - 2 * a3 / a2 + a1 * pgpart - 2 * prob + prob * a3 / a2) / alpha ** 2
        hess_arr[-1, -1] = da2.sum()
        tri_idx = np.triu_indices(dim + 1, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        return hess_arr

    def hessian_factor(self, params):
        """
        Generalized Negative Binomial (NB-P) model hessian maxtrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hessian : ndarray, 2-D
            The hessian matrix of the model.
        """
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        a5 = a4 * p / mu
        dgpart = digamma(a3) - digamma(a1)
        coeff = mu ** 2 * ((1 + a4) ** 2 * a3 / a2 ** 2 - a3 * (a5 - a4 / mu) / a2 - y / mu ** 2 - 2 * a4 * (1 + a4) / a2 + a5 * (np.log(a1) - np.log(a2) + dgpart + 2) - a4 * (np.log(a1) - np.log(a2) + dgpart + 1) / mu - a4 ** 2 * (polygamma(1, a1) - polygamma(1, a3)) + (-(1 + a4) * a3 / a2 + y / mu + a4 * (np.log(a1) - np.log(a2) + dgpart + 1)) / mu)
        hfbb = coeff
        hfba = mu * a1 * ((1 + a4) * (1 - a3 / a2) / a2 - p * (np.log(a1 / a2) + dgpart + 2) / mu + p * (a3 / mu + a4) / a2 + a4 * (polygamma(1, a1) - polygamma(1, a3))) / alpha
        hfaa = a1 * (2 * np.log(a1 / a2) + 2 * dgpart + 3 - 2 * a3 / a2 - a1 * polygamma(1, a1) + a1 * polygamma(1, a3) - 2 * a1 / a2 + a1 * a3 / a2 ** 2) / alpha ** 2
        return (hfbb, hfba, hfaa)

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        mu = const * np.exp(offset + exposure)
        resid = self.endog - mu
        a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
        params.append(a)
        return np.array(params)

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        q = self.parameterization - 1
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((resid ** 2 / mu - 1) * mu ** (-q)).sum() / df_resid
        return a

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, use_transparams=False, cov_type='nonrobust', cov_kwds=None, use_t=None, optim_kwds_prelim=None, **kwargs):
        """
        use_transparams : bool
            This parameter enable internal transformation to impose
            non-negativity. True to enable. Default is False.
            use_transparams=True imposes the no underdispersion (alpha > 0)
            constraint. In case use_transparams=True and method="newton" or
            "ncg" transformation is ignored.
        """
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored', RuntimeWarning)
            self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            kwds_prelim = {'disp': 0, 'skip_hessian': True, 'warn_convergence': False}
            if optim_kwds_prelim is not None:
                kwds_prelim.update(optim_kwds_prelim)
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                res_poi = mod_poi.fit(**kwds_prelim)
            start_params = res_poi.params
            a = self._estimate_dispersion(res_poi.predict(), res_poi.resid, df_resid=res_poi.df_resid)
            start_params = np.append(start_params, max(0.05, a))
        if callback is None:
            callback = lambda *x: x
        mlefit = super().fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, full_output=full_output, callback=callback, **kwargs)
        if optim_kwds_prelim is not None:
            mlefit.mle_settings['optim_kwds_prelim'] = optim_kwds_prelim
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
        nbinfit = NegativeBinomialPResults(self, mlefit._results)
        result = NegativeBinomialPResultsWrapper(nbinfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        _validate_l1_method(method)
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        alpha_p = alpha[:-1] if self.k_extra and np.size(alpha) > 1 else alpha
        self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                start_params = mod_poi.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(start_params, 0.1)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1NegativeBinomialResults(self, cntfit)
        return L1NegativeBinomialResultsWrapper(discretefit)

    @Appender(Poisson.predict.__doc__)
    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        if exog is None:
            exog = self.exog
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)
        if offset is None:
            offset = getattr(self, 'offset', 0)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'var':
            mean = np.exp(linpred)
            alpha = params[-1]
            p = self.parameterization
            var_ = mean * (1 + alpha * mean ** (p - 1))
            return var_
        elif which == 'prob':
            if y_values is None:
                y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog, exposure, offset)
            size, prob = self.convert_params(params, mu)
            return nbinom.pmf(y_values, size[:, None], prob[:, None])
        else:
            raise ValueError('keyword "which" = %s not recognized' % which)

    def convert_params(self, params, mu):
        alpha = params[-1]
        p = 2 - self.parameterization
        size = 1.0 / alpha * mu ** p
        prob = size / (size + mu)
        return (size, prob)

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)
        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2]
        return np.column_stack((d1, d2))

    def _var(self, mu, params=None):
        """variance implied by the distribution

        internal use, will be refactored or removed
        """
        alpha = params[-1]
        p = self.parameterization
        var_ = mu * (1 + alpha * mu ** (p - 1))
        return var_

    def _prob_nonzero(self, mu, params):
        """Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
        alpha = params[-1]
        p = self.parameterization
        prob_nz = 1 - (1 + alpha * mu ** (p - 1)) ** (-1 / alpha)
        return prob_nz

    @Appender(Poisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        """get frozen instance of distribution
        """
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        size, prob = self.convert_params(params, mu)
        distr = nbinom(size, prob)
        return distr