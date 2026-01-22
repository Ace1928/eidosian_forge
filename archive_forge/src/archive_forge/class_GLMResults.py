from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
class GLMResults(base.LikelihoodModelResults):
    """
    Class to contain GLM results.

    GLMResults inherits from statsmodels.LikelihoodModelResults

    Attributes
    ----------
    df_model : float
        See GLM.df_model
    df_resid : float
        See GLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `iterations`,
        `deviance` and `params`.
    model : class instance
        Pointer to GLM model instance that called fit.
    nobs : float
        The number of observations n.
    normalized_cov_params : ndarray
        See GLM docstring
    params : ndarray
        The coefficients of the fitted model.  Note that interpretation
        of the coefficients often depends on the distribution family and the
        data.
    pvalues : ndarray
        The two-tailed p-values for the parameters.
    scale : float
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.
    stand_errors : ndarray
        The standard errors of the fitted GLM.   #TODO still named bse

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale, cov_type='nonrobust', cov_kwds=None, use_t=None):
        super().__init__(model, params, normalized_cov_params=normalized_cov_params, scale=scale)
        self.family = model.family
        self._endog = model.endog
        self.nobs = model.endog.shape[0]
        self._freq_weights = model.freq_weights
        self._var_weights = model.var_weights
        self._iweights = model.iweights
        if isinstance(self.family, families.Binomial):
            self._n_trials = self.model.n_trials
        else:
            self._n_trials = 1
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self._cache = {}
        self._data_attr.extend(['results_constrained', '_freq_weights', '_var_weights', '_iweights'])
        self._data_in_cache.extend(['null', 'mu'])
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.append('mu')
        from statsmodels.base.covtype import get_robustcov_results
        if use_t is None:
            self.use_t = False
        else:
            self.use_t = use_t
        ct = cov_type == 'nonrobust' or cov_type.upper().startswith('HC')
        if self.model._has_freq_weights and (not ct):
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with freq_weights', SpecificationWarning)
        if self.model._has_var_weights and (not ct):
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with var_weights', SpecificationWarning)
        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the' + ' covariance matrix of the errors is correctly ' + 'specified.'}
        else:
            if cov_kwds is None:
                cov_kwds = {}
            get_robustcov_results(self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)

    @cached_data
    def resid_response(self):
        """
        Response residuals.  The response residuals are defined as
        `endog` - `fittedvalues`
        """
        return self._n_trials * (self._endog - self.mu)

    @cached_data
    def resid_pearson(self):
        """
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.families.family and
        statsmodels.families.varfuncs for more information.
        """
        return np.sqrt(self._n_trials) * (self._endog - self.mu) * np.sqrt(self._var_weights) / np.sqrt(self.family.variance(self.mu))

    @cached_data
    def resid_working(self):
        """
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
        """
        val = self.resid_response * self.family.link.deriv(self.mu)
        val *= self._n_trials
        return val

    @cached_data
    def resid_anscombe(self):
        """
        Anscombe residuals.  See statsmodels.families.family for distribution-
        specific Anscombe residuals. Currently, the unscaled residuals are
        provided. In a future version, the scaled residuals will be provided.
        """
        return self.resid_anscombe_scaled

    @cached_data
    def resid_anscombe_scaled(self):
        """
        Scaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        return self.family.resid_anscombe(self._endog, self.fittedvalues, var_weights=self._var_weights, scale=self.scale)

    @cached_data
    def resid_anscombe_unscaled(self):
        """
        Unscaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        return self.family.resid_anscombe(self._endog, self.fittedvalues, var_weights=self._var_weights, scale=1.0)

    @cached_data
    def resid_deviance(self):
        """
        Deviance residuals.  See statsmodels.families.family for distribution-
        specific deviance residuals.
        """
        dev = self.family.resid_dev(self._endog, self.fittedvalues, var_weights=self._var_weights, scale=1.0)
        return dev

    @cached_value
    def pearson_chi2(self):
        """
        Pearson's Chi-Squared statistic is defined as the sum of the squares
        of the Pearson residuals.
        """
        chisq = (self._endog - self.mu) ** 2 / self.family.variance(self.mu)
        chisq *= self._iweights * self._n_trials
        chisqsum = np.sum(chisq)
        return chisqsum

    @cached_data
    def fittedvalues(self):
        """
        The estimated mean response.

        This is the value of the inverse of the link function at
        lin_pred, where lin_pred is the linear predicted value
        obtained by multiplying the design matrix by the coefficient
        vector.
        """
        return self.mu

    @cached_data
    def mu(self):
        """
        See GLM docstring.
        """
        return self.model.predict(self.params)

    @cache_readonly
    def null(self):
        """
        Fitted values of the null model
        """
        endog = self._endog
        model = self.model
        exog = np.ones((len(endog), 1))
        kwargs = model._get_init_kwds().copy()
        kwargs.pop('family')
        for key in getattr(model, '_null_drop_keys', []):
            del kwargs[key]
        start_params = np.atleast_1d(self.family.link(endog.mean()))
        oe = self.model._offset_exposure
        if not (np.size(oe) == 1 and oe == 0):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DomainWarning)
                mod = GLM(endog, exog, family=self.family, **kwargs)
                fitted = mod.fit(start_params=start_params).fittedvalues
        else:
            wls_model = lm.WLS(endog, exog, weights=self._iweights * self._n_trials)
            fitted = wls_model.fit().fittedvalues
        return fitted

    @cache_readonly
    def deviance(self):
        """
        See statsmodels.families.family for the distribution-specific deviance
        functions.
        """
        return self.family.deviance(self._endog, self.mu, self._var_weights, self._freq_weights)

    @cache_readonly
    def null_deviance(self):
        """The value of the deviance function for the model fit with a constant
        as the only regressor."""
        return self.family.deviance(self._endog, self.null, self._var_weights, self._freq_weights)

    @cache_readonly
    def llnull(self):
        """
        Log-likelihood of the model fit with a constant as the only regressor
        """
        return self.family.loglike(self._endog, self.null, var_weights=self._var_weights, freq_weights=self._freq_weights, scale=self.scale)

    def llf_scaled(self, scale=None):
        """
        Return the log-likelihood at the given scale, using the
        estimated scale if the provided scale is None.  In the Gaussian
        case with linear link, the concentrated log-likelihood is
        returned.
        """
        _modelfamily = self.family
        if scale is None:
            if isinstance(self.family, families.Gaussian) and isinstance(self.family.link, families.links.Power) and (self.family.link.power == 1.0):
                scale = (np.power(self._endog - self.mu, 2) * self._iweights).sum()
                scale /= self.model.wnobs
            else:
                scale = self.scale
        val = _modelfamily.loglike(self._endog, self.mu, var_weights=self._var_weights, freq_weights=self._freq_weights, scale=scale)
        return val

    @cached_value
    def llf(self):
        """
        Value of the loglikelihood function evalued at params.
        See statsmodels.families.family for distribution-specific
        loglikelihoods.  The result uses the concentrated
        log-likelihood if the family is Gaussian and the link is linear,
        otherwise it uses the non-concentrated log-likelihood evaluated
        at the estimated scale.
        """
        return self.llf_scaled()

    def pseudo_rsquared(self, kind='cs'):
        """
        Pseudo R-squared

        Cox-Snell likelihood ratio pseudo R-squared is valid for both discrete
        and continuous data. McFadden's pseudo R-squared is only valid for
        discrete data.

        Cox & Snell's pseudo-R-squared:  1 - exp((llnull - llf)*(2/nobs))

        McFadden's pseudo-R-squared: 1 - (llf / llnull)

        Parameters
        ----------
        kind : P"cs", "mcf"}
            Type of pseudo R-square to return

        Returns
        -------
        float
            Pseudo R-squared
        """
        kind = kind.lower()
        if kind.startswith('mcf'):
            prsq = 1 - self.llf / self.llnull
        elif kind.startswith('cox') or kind in ['cs', 'lr']:
            prsq = 1 - np.exp((self.llnull - self.llf) * (2 / self.nobs))
        else:
            raise ValueError('only McFadden and Cox-Snell are available')
        return prsq

    @cached_value
    def aic(self):
        """
        Akaike Information Criterion
        -2 * `llf` + 2 * (`df_model` + 1)
        """
        return self.info_criteria('aic')

    @property
    def bic(self):
        """
        Bayes Information Criterion

        `deviance` - `df_resid` * log(`nobs`)

        .. warning::

            The current definition is based on the deviance rather than the
            log-likelihood. This is not consistent with the AIC definition,
            and after 0.13 both will make use of the log-likelihood definition.

        Notes
        -----
        The log-likelihood version is defined
        -2 * `llf` + (`df_model` + 1)*log(n)
        """
        if _use_bic_helper.use_bic_llf not in (True, False):
            warnings.warn('The bic value is computed using the deviance formula. After 0.13 this will change to the log-likelihood based formula. This change has no impact on the relative rank of models compared using BIC. You can directly access the log-likelihood version using the `bic_llf` attribute. You can suppress this message by calling statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF with True to get the LLF-based version now or False to retainthe deviance version.', FutureWarning)
        if bool(_use_bic_helper.use_bic_llf):
            return self.bic_llf
        return self.bic_deviance

    @cached_value
    def bic_deviance(self):
        """
        Bayes Information Criterion

        Based on the deviance,
        `deviance` - `df_resid` * log(`nobs`)
        """
        return self.deviance - (self.model.wnobs - self.df_model - 1) * np.log(self.model.wnobs)

    @cached_value
    def bic_llf(self):
        """
        Bayes Information Criterion

        Based on the log-likelihood,
        -2 * `llf` + log(n) * (`df_model` + 1)
        """
        return self.info_criteria('bic')

    def info_criteria(self, crit, scale=None, dk_params=0):
        """Return an information criterion for the model.

        Parameters
        ----------
        crit : string
            One of 'aic', 'bic', or 'qaic'.
        scale : float
            The scale parameter estimated using the parent model,
            used only for qaic.
        dk_params : int or float
            Correction to the number of parameters used in the information
            criterion. By default, only mean parameters are included, the
            scale parameter is not included in the parameter count.
            Use ``dk_params=1`` to include scale in the parameter count.

        Returns
        -------
        Value of information criterion.

        Notes
        -----
        The quasi-Akaike Information criterion (qaic) is -2 *
        `llf`/`scale` + 2 * (`df_model` + 1).  It may not give
        meaningful results except for Poisson and related models.

        The QAIC (ic_type='qaic') must be evaluated with a provided
        scale parameter.  Two QAIC values are only comparable if they
        are calculated using the same scale parameter.  The scale
        parameter should be estimated using the largest model among
        all models being compared.

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        crit = crit.lower()
        k_params = self.df_model + 1 + dk_params
        if crit == 'aic':
            return -2 * self.llf + 2 * k_params
        elif crit == 'bic':
            nobs = self.df_model + self.df_resid + 1
            bic = -2 * self.llf + k_params * np.log(nobs)
            return bic
        elif crit == 'qaic':
            f = self.model.family
            fl = (families.Poisson, families.NegativeBinomial, families.Binomial)
            if not isinstance(f, fl):
                msg = 'QAIC is only valid for Binomial, Poisson and '
                msg += 'Negative Binomial families.'
                warnings.warn(msg)
            llf = self.llf_scaled(scale=1)
            return -2 * llf / scale + 2 * k_params

    def get_prediction(self, exog=None, exposure=None, offset=None, transform=True, which=None, linear=None, average=False, agg_weights=None, row_labels=None):
        """
    Compute prediction results for GLM compatible models.

    Options and return class depend on whether "which" is None or not.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    exposure : array_like, optional
        Exposure time values, only can be used with the log link
        function.
    offset : array_like, optional
        Offset values.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    which : 'mean', 'linear', 'var'(optional)
        Statitistic to predict. Default is 'mean'.
        If which is None, then the deprecated keyword "linear" applies.
        If which is not None, then a generic Prediction results class will
        be returned. Some options are only available if which is not None.
        See notes.

        - 'mean' returns the conditional expectation of endog E(y | x),
          i.e. inverse of the model's link function of linear predictor.
        - 'linear' returns the linear predictor of the mean function.
        - 'var_unscaled' variance of endog implied by the likelihood model.
          This does not include scale or var_weights.

    linear : bool
        The ``linear` keyword is deprecated and will be removed,
        use ``which`` keyword instead.
        If which is None, then the linear keyword is used, otherwise it will
        be ignored.
        If True and which is None, the linear predicted values are returned.
        If False or None, then the statistic specified by ``which`` will be
        returned.
    average : bool
        Keyword is only used if ``which`` is not None.
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then the average
        over observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Keyword is only used if ``which`` is not None.
        Aggregation weights, only used if average is True.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    prediction_results : instance of a PredictionResults class.
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
        The Results class of the return depends on the value of ``which``.

    See Also
    --------
    GLM.predict
    GLMResults.predict

    Notes
    -----
    Changes in statsmodels 0.14: The ``which`` keyword has been added.
    If ``which`` is None, then the behavior is the same as in previous
    versions, and returns the mean and linear prediction results.
    If the ``which`` keyword is not None, then a generic prediction results
    class is returned and is not backwards compatible with the old prediction
    results class, e.g. column names of summary_frame differs.
    There are more choices for the returned predicted statistic using
    ``which``. More choices will be added in the next release.
    Two additional keyword, average and agg_weights options are now also
    available if ``which`` is not None.
    In a future version ``which`` will become not None and the backwards
    compatible prediction results class will be removed.

    """
        import statsmodels.regression._prediction as linpred
        pred_kwds = {'exposure': exposure, 'offset': offset, 'which': 'linear'}
        if which is None:
            res_linpred = linpred.get_prediction(self, exog=exog, transform=transform, row_labels=row_labels, pred_kwds=pred_kwds)
            pred_kwds['which'] = 'mean'
            res = pred.get_prediction_glm(self, exog=exog, transform=transform, row_labels=row_labels, linpred=res_linpred, link=self.model.family.link, pred_kwds=pred_kwds)
        else:
            pred_kwds = {'exposure': exposure, 'offset': offset}
            res = pred.get_prediction(self, exog=exog, which=which, transform=transform, row_labels=row_labels, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
        return res

    @Appender(pinfer.score_test.__doc__)
    def score_test(self, exog_extra=None, params_constrained=None, hypothesis='joint', cov_type=None, cov_kwds=None, k_constraints=None, observed=True):
        if self.model._has_freq_weights is True:
            warnings.warn('score test has not been verified with freq_weights', UserWarning)
        if self.model._has_var_weights is True:
            warnings.warn('score test has not been verified with var_weights', UserWarning)
        mod_df_resid = self.model.df_resid
        self.model.df_resid = self.df_resid
        if k_constraints is not None:
            self.model.df_resid += k_constraints
        res = pinfer.score_test(self, exog_extra=exog_extra, params_constrained=params_constrained, hypothesis=hypothesis, cov_type=cov_type, cov_kwds=cov_kwds, k_constraints=k_constraints, scale=None, observed=observed)
        self.model.df_resid = mod_df_resid
        return res

    def get_hat_matrix_diag(self, observed=True):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        weights = self.model.hessian_factor(self.params, observed=observed)
        wexog = np.sqrt(weights)[:, None] * self.model.exog
        hd = (wexog * np.linalg.pinv(wexog).T).sum(1)
        return hd

    def get_influence(self, observed=True):
        """
        Get an instance of GLMInfluence with influence and outlier measures

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        infl : GLMInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.GLMInfluence
        """
        from statsmodels.stats.outliers_influence import GLMInfluence
        weights = self.model.hessian_factor(self.params, observed=observed)
        weights_sqrt = np.sqrt(weights)
        wexog = weights_sqrt[:, None] * self.model.exog
        wendog = weights_sqrt * self.model.endog
        hat_matrix_diag = self.get_hat_matrix_diag(observed=observed)
        infl = GLMInfluence(self, endog=wendog, exog=wexog, resid=self.resid_pearson / np.sqrt(self.scale), hat_matrix_diag=hat_matrix_diag)
        return infl

    def get_distribution(self, exog=None, exposure=None, offset=None, var_weights=1.0, n_trials=1.0):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        scale : scalar
            The scale parameter.
        exog : array_like
            The predictor variable matrix.
        offset : array_like or None
            Offset variable for predicted mean.
        exposure : array_like or None
            Log(exposure) will be added to the linear prediction.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is None.
        n_trials : int
            Number of trials for the binomial distribution. The default is 1
            which corresponds to a Bernoulli random variable.

        Returns
        -------
        gen
            Instance of a scipy frozen distribution based on estimated
            parameters.
            Use the ``rvs`` method to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        if isinstance(self.model.family, (families.Binomial, families.Poisson, families.NegativeBinomial)):
            scale = 1.0
            if self.scale != 1.0:
                msg = 'using scale=1, no exess dispersion in distribution'
                warnings.warn(msg, UserWarning)
        else:
            scale = self.scale
        mu = self.predict(exog, exposure, offset, which='mean')
        kwds = {}
        if np.any(n_trials != 1) and isinstance(self.model.family, families.Binomial):
            kwds['n_trials'] = n_trials
        distr = self.model.family.get_distribution(mu, scale, var_weights=var_weights, **kwds)
        return distr

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Warning: offset, exposure and weights (var_weights and freq_weights)
        are not supported by margeff.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables. For interpretations of these methods
            see notes below.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        Interpretations of methods:

        - 'dydx' - change in `endog` for a change in `exog`.
        - 'eyex' - proportional change in `endog` for a proportional change
          in `exog`.
        - 'dyex' - change in `endog` for a proportional change in `exog`.
        - 'eydx' - proportional change in `endog` for a change in `exog`.

        When using after Poisson, returns the expected number of events per
        period, assuming that the model is loglinear.

        Status : unsupported features offset, exposure and weights. Default
        handling of freq_weights for average effect "overall" might change.

        """
        if getattr(self.model, 'offset', None) is not None:
            raise NotImplementedError('Margins with offset are not available.')
        if np.any(self.model.var_weights != 1) or np.any(self.model.freq_weights != 1):
            warnings.warn('weights are not taken into account by margeff')
        from statsmodels.discrete.discrete_margins import DiscreteMargins
        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    @Appender(base.LikelihoodModelResults.remove_data.__doc__)
    def remove_data(self):
        self._data_attr.extend([i for i in self.model._data_attr if '_data.' not in i])
        super(self.__class__, self).remove_data()
        self._endog = None
        self._freq_weights = None
        self._var_weights = None
        self._iweights = None
        self._n_trials = None

    @Appender(_plot_added_variable_doc % {'extra_params_doc': ''})
    def plot_added_variable(self, focus_exog, resid_type=None, use_glm_weights=True, fit_kwargs=None, ax=None):
        from statsmodels.graphics.regressionplots import plot_added_variable
        fig = plot_added_variable(self, focus_exog, resid_type=resid_type, use_glm_weights=use_glm_weights, fit_kwargs=fit_kwargs, ax=ax)
        return fig

    @Appender(_plot_partial_residuals_doc % {'extra_params_doc': ''})
    def plot_partial_residuals(self, focus_exog, ax=None):
        from statsmodels.graphics.regressionplots import plot_partial_residuals
        return plot_partial_residuals(self, focus_exog, ax=ax)

    @Appender(_plot_ceres_residuals_doc % {'extra_params_doc': ''})
    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None, ax=None):
        from statsmodels.graphics.regressionplots import plot_ceres_residuals
        return plot_ceres_residuals(self, focus_exog, frac, cond_means=cond_means, ax=ax)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
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
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Model Family:', [self.family.__class__.__name__]), ('Link Function:', [self.family.link.__class__.__name__]), ('Method:', [self.method]), ('Date:', None), ('Time:', None), ('No. Iterations:', ['%d' % self.fit_history['iteration']])]
        try:
            prsquared = self.pseudo_rsquared(kind='cs')
        except ValueError:
            prsquared = np.nan
        top_right = [('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None), ('Scale:', ['%#8.5g' % self.scale]), ('Log-Likelihood:', None), ('Deviance:', ['%#8.5g' % self.deviance]), ('Pearson chi2:', ['%#6.3g' % self.pearson_chi2]), ('Pseudo R-squ. (CS):', ['%#6.4g' % prsquared])]
        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))
        if title is None:
            title = 'Generalized Linear Model Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear equality constraints.'])
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format='%.4f'):
        """Experimental summary for regression Results

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        self.method = 'IRLS'
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            smry.add_base(results=self, alpha=alpha, float_format=float_format, xname=xname, yname=yname, title=title)
        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear equality constraints.')
        return smry