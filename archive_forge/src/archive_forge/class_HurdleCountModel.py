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
class HurdleCountModel(CountModel):
    __doc__ = "\n    Hurdle model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    dist : string\n        Log-likelihood type of count model family. 'poisson' or 'negbin'\n    zerodist : string\n        Log-likelihood type of zero hurdle model family. 'poisson', 'negbin'\n    p : scalar\n        Define parameterization for count model.\n        Used when dist='negbin'.\n    pzero : scalar\n        Define parameterization parameter zero hurdle model family.\n        Used when zerodist='negbin'.\n    " % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    Notes\n    -----\n    The parameters in the NegativeBinomial zero model are not identified if\n    the predicted mean is constant. If there is no or only little variation in\n    the predicted mean, then convergence might fail, hessian might not be\n    invertible or parameter estimates will have large standard errors.\n\n    References\n    ----------\n    not yet\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, dist='poisson', zerodist='poisson', p=2, pzero=2, exposure=None, missing='none', **kwargs):
        if offset is not None or exposure is not None:
            msg = 'Offset and exposure are not yet implemented'
            raise NotImplementedError(msg)
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.k_extra1 = 0
        self.k_extra2 = 0
        self._initialize(dist, zerodist, p, pzero)
        self.result_class = HurdleCountResults
        self.result_class_wrapper = HurdleCountResultsWrapper
        self.result_class_reg = L1HurdleCountResults
        self.result_class_reg_wrapper = L1HurdleCountResultsWrapper

    def _initialize(self, dist, zerodist, p, pzero):
        if dist not in ['poisson', 'negbin'] or zerodist not in ['poisson', 'negbin']:
            raise NotImplementedError('dist and zerodist must be "poisson","negbin"')
        if zerodist == 'poisson':
            self.model1 = _RCensored(self.endog, self.exog, model=Poisson)
        elif zerodist == 'negbin':
            self.model1 = _RCensored(self.endog, self.exog, model=NegativeBinomialP)
            self.k_extra1 += 1
        if dist == 'poisson':
            self.model2 = TruncatedLFPoisson(self.endog, self.exog)
        elif dist == 'negbin':
            self.model2 = TruncatedLFNegativeBinomialP(self.endog, self.exog, p=p)
            self.k_extra2 += 1

    def loglike(self, params):
        """
        Loglikelihood of Generic Hurdle model

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
        k = int((len(params) - self.k_extra1 - self.k_extra2) / 2) + self.k_extra1
        return self.model1.loglike(params[:k]) + self.model2.loglike(params[k:])

    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if cov_type != 'nonrobust':
            raise ValueError('robust cov_type currently not supported')
        results1 = self.model1.fit(start_params=start_params, method=method, maxiter=maxiter, disp=disp, full_output=full_output, callback=lambda x: x, **kwargs)
        results2 = self.model2.fit(start_params=start_params, method=method, maxiter=maxiter, disp=disp, full_output=full_output, callback=lambda x: x, **kwargs)
        result = deepcopy(results1)
        result._results.model = self
        result.mle_retvals['converged'] = [results1.mle_retvals['converged'], results2.mle_retvals['converged']]
        result._results.params = np.append(results1._results.params, results2._results.params)
        result._results.df_model += results2._results.df_model
        self.k_extra1 += getattr(results1._results, 'k_extra', 0)
        self.k_extra2 += getattr(results2._results, 'k_extra', 0)
        self.k_extra = self.k_extra1 + self.k_extra2 + 1
        xnames1 = ['zm_' + name for name in self.model1.exog_names]
        self.exog_names[:] = xnames1 + self.model2.exog_names
        from scipy.linalg import block_diag
        result._results.normalized_cov_params = None
        try:
            cov1 = results1._results.cov_params()
            cov2 = results2._results.cov_params()
            result._results.normalized_cov_params = block_diag(cov1, cov2)
        except ValueError as e:
            if 'need covariance' not in str(e):
                raise
        modelfit = self.result_class(self, result._results, results1, results2)
        result = self.result_class_wrapper(modelfit)
        return result
    fit.__doc__ = DiscreteModel.fit.__doc__

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
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
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
            - 'prob-main' : probability of selecting the main model which is
              the probability of observing a nonzero count P(y > 0 | x).
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x).
              This is equal to is ``1 - prob-main``
            - 'prob-trunc' : probability of truncation of the truncated count
              model. This is the probability of observing a zero count implied
              by the truncation model.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob' : probabilities of each count from 0 to max(endog), or
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
        'prob-zero' / 'prob-trunc' is the ratio of probabilities of observing
        a zero count between hurdle model and the truncated count model.
        If this ratio is larger than one, then the hurdle model has an inflated
        number of zeros compared to the count model. If it is smaller than one,
        then the number of zeros is deflated.
        """
        which = which.lower()
        no_exog = True if exog is None else False
        exog, offset, exposure = self._get_predict_arrays(exog=exog, offset=offset, exposure=exposure)
        exog_zero = None
        if exog_zero is None:
            if no_exog:
                exog_zero = self.exog
            else:
                exog_zero = exog
        k_zeros = int((len(params) - self.k_extra1 - self.k_extra2) / 2) + self.k_extra1
        params_zero = params[:k_zeros]
        params_main = params[k_zeros:]
        lin_pred = np.dot(exog, params_main[:self.exog.shape[1]]) + exposure + offset
        mu1 = self.model1.predict(params_zero, exog=exog)
        prob_main = self.model1.model_main._prob_nonzero(mu1, params_zero)
        prob_zero = 1 - prob_main
        mu2 = np.exp(lin_pred)
        prob_ntrunc = self.model2.model_main._prob_nonzero(mu2, params_main)
        if which == 'mean':
            return prob_main * np.exp(lin_pred) / prob_ntrunc
        elif which == 'mean-main':
            return np.exp(lin_pred)
        elif which == 'linear':
            return lin_pred
        elif which == 'mean-nonzero':
            return np.exp(lin_pred) / prob_ntrunc
        elif which == 'prob-zero':
            return prob_zero
        elif which == 'prob-main':
            return prob_main
        elif which == 'prob-trunc':
            return 1 - prob_ntrunc
        elif which == 'var':
            mu = np.exp(lin_pred)
            mt, vt = self.model2._predict_mom_trunc0(params_main, mu)
            var_ = prob_main * vt + prob_main * (1 - prob_main) * mt ** 2
            return var_
        elif which == 'prob':
            probs_main = self.model2.predict(params_main, exog, np.exp(exposure), offset, which='prob', y_values=y_values)
            probs_main *= prob_main[:, None]
            probs_main[:, 0] = prob_zero
            return probs_main
        else:
            raise ValueError('which = %s is not available' % which)