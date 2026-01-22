from collections import OrderedDict
import contextlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.base.data import PandasData
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
from statsmodels.tools.sm_exceptions import PrecisionWarning
from statsmodels.tools.numdiff import (
from statsmodels.tools.tools import pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.statespace.tools import _safe_cond
class StateSpaceMLEResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : ndarray
        Fitted parameters

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    """

    def __init__(self, model, params, scale=1.0):
        self.data = model.data
        self.endog = model.data.orig_endog
        super().__init__(model, params, None, scale=scale)
        self._has_fixed_params = self.model._has_fixed_params
        self._fixed_params_index = self.model._fixed_params_index
        self._free_params_index = self.model._free_params_index
        if self._has_fixed_params:
            self._fixed_params = self.model._fixed_params.copy()
            self.fixed_params = list(self._fixed_params.keys())
        else:
            self._fixed_params = None
            self.fixed_params = []
        self.param_names = ['%s (fixed)' % name if name in self.fixed_params else name for name in self.data.param_names or []]
        self.nobs = self.model.nobs
        self.k_params = self.model.k_params
        self._rank = None

    @cache_readonly
    def nobs_effective(self):
        raise NotImplementedError

    @cache_readonly
    def df_resid(self):
        return self.nobs_effective - self.df_model

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        return aic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def aicc(self):
        """
        (float) Akaike Information Criterion with small sample correction
        """
        return aicc(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        return bic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def fittedvalues(self):
        raise NotImplementedError

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        return hqic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        raise NotImplementedError

    @cache_readonly
    def mae(self):
        """
        (float) Mean absolute error
        """
        return np.mean(np.abs(self.resid))

    @cache_readonly
    def mse(self):
        """
        (float) Mean squared error
        """
        return self.sse / self.nobs

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        pvalues = np.zeros_like(self.zvalues) * np.nan
        mask = np.ones_like(pvalues, dtype=bool)
        mask[self._free_params_index] = True
        mask &= ~np.isnan(self.zvalues)
        pvalues[mask] = norm.sf(np.abs(self.zvalues[mask])) * 2
        return pvalues

    @cache_readonly
    def resid(self):
        raise NotImplementedError

    @cache_readonly
    def sse(self):
        """
        (float) Sum of squared errors
        """
        return np.sum(self.resid ** 2)

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def _get_prediction_start_index(self, anchor):
        """Returns a valid numeric start index for predictions/simulations"""
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start
            iloc += 1
        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation outside of the sample.')
        return iloc

    def _cov_params_approx(self, approx_complex_step=True, approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model.hessian(params=self.params, transformed=True, includes_fixed=True, method='approx', approx_complex_step=approx_complex_step, approx_centered=approx_centered)
        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            if len(self.fixed_params) < self.k_params:
                tmp, singular_values = pinv_extended(evaluated_hessian[mask])
            else:
                tmp, singular_values = (np.nan, [np.nan])
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            neg_cov[mask] = tmp
        else:
            neg_cov, singular_values = pinv_extended(evaluated_hessian)
        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        return self._cov_params_approx(self._cov_approx_complex_step, self._cov_approx_centered)

    def test_serial_correlation(self, method, lags=None):
        """
        Ljung-Box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        method : {'ljungbox', 'boxpierce', None}
            The statistical test for serial correlation. If None, an attempt is
            made to select an appropriate test.
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is min(10, nobs//5) for
            non-seasonal time series and min (2*m, nobs//5) for seasonal time
            series.

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable and each lag. The array is then sized
            `(k_endog, 2, lags)`. If the method is called as
            `ljungbox = res.test_serial_correlation()`, then `ljungbox[i]`
            holds the results of the Ljung-Box test (as would be returned by
            `statsmodels.stats.diagnostic.acorr_ljungbox`) for the `i` th
            endogenous variable.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox
            Ljung-Box test for serial correlation.

        Notes
        -----
        For statespace models: let `d` = max(loglikelihood_burn, nobs_diffuse);
        this test is calculated ignoring the first `d` residuals.

        Output is nan for any endogenous variable which has missing values.
        """
        if method is None:
            method = 'ljungbox'
        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized forecast errors have not been computed.')
        if method == 'ljungbox' or method == 'boxpierce':
            from statsmodels.stats.diagnostic import acorr_ljungbox
            if hasattr(self, 'loglikelihood_burn'):
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
                nobs_effective = self.nobs - d
            else:
                nobs_effective = self.nobs_effective
            output = []
            if lags is None:
                seasonal_periods = getattr(self.model, 'seasonal_periods', 0)
                if seasonal_periods:
                    lags = min(2 * seasonal_periods, nobs_effective // 5)
                else:
                    lags = min(10, nobs_effective // 5)
            cols = [2, 3] if method == 'boxpierce' else [0, 1]
            for i in range(self.model.k_endog):
                if hasattr(self, 'filter_results'):
                    x = self.filter_results.standardized_forecasts_error[i][d:]
                else:
                    x = self.standardized_forecasts_error
                results = acorr_ljungbox(x, lags=lags, boxpierce=method == 'boxpierce')
                output.append(np.asarray(results)[:, cols].T)
            output = np.c_[output]
        else:
            raise NotImplementedError('Invalid serial correlation test method.')
        return output

    def test_heteroskedasticity(self, method, alternative='two-sided', use_f=True):
        """
        Test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
        is of no heteroskedasticity.

        Parameters
        ----------
        method : {'breakvar', None}
            The statistical test for heteroskedasticity. Must be 'breakvar'
            for test of a break in the variance. If None, an attempt is
            made to select an appropriate test.
        alternative : str, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        use_f : bool, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is True (i.e. default is to compare against an F
            distribution).

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        Notes
        -----
        The null hypothesis is of no heteroskedasticity. That means different
        things depending on which alternative is selected:

        - Increasing: Null hypothesis is that the variance is not increasing
          throughout the sample; that the sum-of-squares in the later
          subsample is *not* greater than the sum-of-squares in the earlier
          subsample.
        - Decreasing: Null hypothesis is that the variance is not decreasing
          throughout the sample; that the sum-of-squares in the earlier
          subsample is *not* greater than the sum-of-squares in the later
          subsample.
        - Two-sided: Null hypothesis is that the variance is not changing
          throughout the sample. Both that the sum-of-squares in the earlier
          subsample is not greater than the sum-of-squares in the later
          subsample *and* that the sum-of-squares in the later subsample is
          not greater than the sum-of-squares in the earlier subsample.

        For :math:`h = [T/3]`, the test statistic is:

        .. math::

            H(h) = \\sum_{t=T-h+1}^T  \\tilde v_t^2
            \\Bigg / \\sum_{t=d+1}^{d+1+h} \\tilde v_t^2

        where :math:`d` = max(loglikelihood_burn, nobs_diffuse)` (usually
        corresponding to diffuse initialization under either the approximate
        or exact approach).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\\chi_h^2`; this second test can be applied by passing
        `asymptotic=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        References
        ----------
        .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
               *Models and the Kalman Filter.* Cambridge University Press.
        """
        if method is None:
            method = 'breakvar'
        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized forecast errors have not been computed.')
        if method == 'breakvar':
            if hasattr(self, 'filter_results'):
                squared_resid = self.filter_results.standardized_forecasts_error ** 2
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
                nobs_effective = self.nobs - d
            else:
                squared_resid = self.standardized_forecasts_error ** 2
                if squared_resid.ndim == 1:
                    squared_resid = np.asarray(squared_resid)
                    squared_resid = squared_resid[np.newaxis, :]
                nobs_effective = self.nobs_effective
                d = 0
            squared_resid = np.asarray(squared_resid)
            test_statistics = []
            p_values = []
            for i in range(self.model.k_endog):
                h = int(np.round(nobs_effective / 3))
                numer_resid = squared_resid[i, -h:]
                numer_resid = numer_resid[~np.isnan(numer_resid)]
                numer_dof = len(numer_resid)
                denom_resid = squared_resid[i, d:d + h]
                denom_resid = denom_resid[~np.isnan(denom_resid)]
                denom_dof = len(denom_resid)
                if numer_dof < 2:
                    warnings.warn('Early subset of data for variable %d  has too few non-missing observations to calculate test statistic.' % i, stacklevel=2)
                    numer_resid = np.nan
                if denom_dof < 2:
                    warnings.warn('Later subset of data for variable %d  has too few non-missing observations to calculate test statistic.' % i, stacklevel=2)
                    denom_resid = np.nan
                test_statistic = np.sum(numer_resid) / np.sum(denom_resid)
                if use_f:
                    from scipy.stats import f
                    pval_lower = lambda test_statistics: f.cdf(test_statistics, numer_dof, denom_dof)
                    pval_upper = lambda test_statistics: f.sf(test_statistics, numer_dof, denom_dof)
                else:
                    from scipy.stats import chi2
                    pval_lower = lambda test_statistics: chi2.cdf(numer_dof * test_statistics, denom_dof)
                    pval_upper = lambda test_statistics: chi2.sf(numer_dof * test_statistics, denom_dof)
                alternative = alternative.lower()
                if alternative in ['i', 'inc', 'increasing']:
                    p_value = pval_upper(test_statistic)
                elif alternative in ['d', 'dec', 'decreasing']:
                    test_statistic = 1.0 / test_statistic
                    p_value = pval_upper(test_statistic)
                elif alternative in ['2', '2-sided', 'two-sided']:
                    p_value = 2 * np.minimum(pval_lower(test_statistic), pval_upper(test_statistic))
                else:
                    raise ValueError('Invalid alternative.')
                test_statistics.append(test_statistic)
                p_values.append(p_value)
            output = np.c_[test_statistics, p_values]
        else:
            raise NotImplementedError('Invalid heteroskedasticity test method.')
        return output

    def test_normality(self, method):
        """
        Test for normality of standardized residuals.

        Null hypothesis is normality.

        Parameters
        ----------
        method : {'jarquebera', None}
            The statistical test for normality. Must be 'jarquebera' for
            Jarque-Bera normality test. If None, an attempt is made to select
            an appropriate test.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera
            The Jarque-Bera test of normality.

        Notes
        -----
        For statespace models: let `d` = max(loglikelihood_burn, nobs_diffuse);
        this test is calculated ignoring the first `d` residuals.

        In the case of missing data, the maintained hypothesis is that the
        data are missing completely at random. This test is then run on the
        standardized residuals excluding those corresponding to missing
        observations.
        """
        if method is None:
            method = 'jarquebera'
        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized forecast errors have not been computed.')
        if method == 'jarquebera':
            from statsmodels.stats.stattools import jarque_bera
            if hasattr(self, 'loglikelihood_burn'):
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            else:
                d = 0
            output = []
            for i in range(self.model.k_endog):
                if hasattr(self, 'fiter_results'):
                    resid = self.filter_results.standardized_forecasts_error[i, d:]
                else:
                    resid = self.standardized_forecasts_error
                mask = ~np.isnan(resid)
                output.append(jarque_bera(resid[mask]))
        else:
            raise NotImplementedError('Invalid normality test method.')
        return np.array(output)

    def summary(self, alpha=0.05, start=None, title=None, model_name=None, display_params=True):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        model_name : str
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        model = self.model
        if title is None:
            title = 'Statespace Model Results'
        if start is None:
            start = 0
        if self.model._index_dates:
            ix = self.model._index
            d = ix[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.nobs)]
        if model_name is None:
            model_name = model.__class__.__name__
        try:
            het = self.test_heteroskedasticity(method='breakvar')
        except Exception:
            het = np.array([[np.nan] * 2])
        try:
            lb = self.test_serial_correlation(method='ljungbox')
        except Exception:
            lb = np.array([[np.nan] * 2]).reshape(1, 2, 1)
        try:
            jb = self.test_normality(method='jarquebera')
        except Exception:
            jb = np.array([[np.nan] * 4])
        if not isinstance(model_name, list):
            model_name = [model_name]
        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [('Date:', None), ('Time:', None), ('Sample:', [sample[0]]), ('', [sample[1]])]
        top_right = [('No. Observations:', [self.nobs]), ('Log Likelihood', ['%#5.3f' % self.llf])]
        if hasattr(self, 'rsquared'):
            top_right.append(('R-squared:', ['%#8.3f' % self.rsquared]))
        top_right += [('AIC', ['%#5.3f' % self.aic]), ('BIC', ['%#5.3f' % self.bic]), ('HQIC', ['%#5.3f' % self.hqic])]
        if hasattr(self, 'filter_results'):
            if self.filter_results is not None and self.filter_results.filter_concentrated:
                top_right.append(('Scale', ['%#5.3f' % self.scale]))
        else:
            top_right.append(('Scale', ['%#5.3f' % self.scale]))
        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))
        format_str = lambda array: [', '.join([f'{i:.2f}' for i in array])]
        diagn_left = [('Ljung-Box (Q):', format_str(lb[:, 0, -1])), ('Prob(Q):', format_str(lb[:, 1, -1])), ('Heteroskedasticity (H):', format_str(het[:, 0])), ('Prob(H) (two-sided):', format_str(het[:, 1]))]
        diagn_right = [('Jarque-Bera (JB):', format_str(jb[:, 0])), ('Prob(JB):', format_str(jb[:, 1])), ('Skew:', format_str(jb[:, 2])), ('Kurtosis:', format_str(jb[:, 3]))]
        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)
        if len(self.params) > 0 and display_params:
            summary.add_table_params(self, alpha=alpha, xname=self.param_names, use_t=False)
        summary.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, title='')
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < len(self.params) - len(self.fixed_params):
            cov_params = self.cov_params()
            if len(self.fixed_params) > 0:
                mask = np.ix_(self._free_params_index, self._free_params_index)
                cov_params = cov_params[mask]
            etext.append('Covariance matrix is singular or near-singular, with condition number %6.3g. Standard errors may be unstable.' % _safe_cond(cov_params))
        if etext:
            etext = [f'[{i + 1}] {text}' for i, text in enumerate(etext)]
            etext.insert(0, 'Warnings:')
            summary.add_extra_txt(etext)
        return summary