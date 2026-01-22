import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
class RecursiveLSResults(MLEResults):
    """
    Class to hold results from fitting a recursive least squares model.

    Parameters
    ----------
    model : RecursiveLS instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the recursive least squares
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg', **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        q = max(self.loglikelihood_burn, self.k_diffuse_states)
        self.df_model = q - self.model.k_constraints
        self.df_resid = self.nobs_effective - self.df_model
        self._init_kwds = self.model._get_init_kwds()
        self.specification = Bunch(**{'k_exog': self.model.k_exog, 'k_constraints': self.model.k_constraints})
        if self.model._r_matrix is not None:
            for name in ['forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov']:
                setattr(self, name, getattr(self, name)[0:1])

    @property
    def recursive_coefficients(self):
        """
        Estimates of regression coefficients, recursively estimated

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        out = None
        spec = self.specification
        start = offset = 0
        end = offset + spec.k_exog
        out = Bunch(filtered=self.filtered_state[start:end], filtered_cov=self.filtered_state_cov[start:end, start:end], smoothed=None, smoothed_cov=None, offset=offset)
        if self.smoothed_state is not None:
            out.smoothed = self.smoothed_state[start:end]
        if self.smoothed_state_cov is not None:
            out.smoothed_cov = self.smoothed_state_cov[start:end, start:end]
        return out

    @cache_readonly
    def resid_recursive(self):
        """
        Recursive residuals

        Returns
        -------
        resid_recursive : array_like
            An array of length `nobs` holding the recursive
            residuals.

        Notes
        -----
        These quantities are defined in, for example, Harvey (1989)
        section 5.4. In fact, there he defines the standardized innovations in
        equation 5.4.1, but in his version they have non-unit variance, whereas
        the standardized forecast errors computed by the Kalman filter here
        assume unit variance. To convert to Harvey's definition, we need to
        multiply by the standard deviation.

        Harvey notes that in smaller samples, "although the second moment
        of the :math:`\\tilde \\sigma_*^{-1} \\tilde v_t`'s is unity, the
        variance is not necessarily equal to unity as the mean need not be
        equal to zero", and he defines an alternative version (which are
        not provided here).
        """
        return self.filter_results.standardized_forecasts_error[0] * self.scale ** 0.5

    @cache_readonly
    def cusum(self):
        """
        Cumulative sum of standardized recursive residuals statistics

        Returns
        -------
        cusum : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM statistics.

        Notes
        -----
        The CUSUM statistic takes the form:

        .. math::

            W_t = \\frac{1}{\\hat \\sigma} \\sum_{j=k+1}^t w_j

        where :math:`w_j` is the recursive residual at time :math:`j` and
        :math:`\\hat \\sigma` is the estimate of the standard deviation
        from the full sample.

        Excludes the first `k_exog` datapoints.

        Due to differences in the way :math:`\\hat \\sigma` is calculated, the
        output of this function differs slightly from the output in the
        R package strucchange and the Stata contributed .ado file cusum6. The
        calculation in this package is consistent with the description of
        Brown et al. (1975)

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        return np.cumsum(self.resid_recursive[d:]) / np.std(self.resid_recursive[d:], ddof=1)

    @cache_readonly
    def cusum_squares(self):
        """
        Cumulative sum of squares of standardized recursive residuals
        statistics

        Returns
        -------
        cusum_squares : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM of squares statistics.

        Notes
        -----
        The CUSUM of squares statistic takes the form:

        .. math::

            s_t = \\left ( \\sum_{j=k+1}^t w_j^2 \\right ) \\Bigg /
                  \\left ( \\sum_{j=k+1}^T w_j^2 \\right )

        where :math:`w_j` is the recursive residual at time :math:`j`.

        Excludes the first `k_exog` datapoints.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        numer = np.cumsum(self.resid_recursive[d:] ** 2)
        denom = numer[-1]
        return numer / denom

    @cache_readonly
    def llf_recursive_obs(self):
        """
        (float) Loglikelihood at observation, computed from recursive residuals
        """
        from scipy.stats import norm
        return np.log(norm.pdf(self.resid_recursive, loc=0, scale=self.scale ** 0.5))

    @cache_readonly
    def llf_recursive(self):
        """
        (float) Loglikelihood defined by recursive residuals, equivalent to OLS
        """
        return np.sum(self.llf_recursive_obs)

    @cache_readonly
    def ssr(self):
        """ssr"""
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        return (self.nobs - d) * self.filter_results.obs_cov[0, 0, 0]

    @cache_readonly
    def centered_tss(self):
        """Centered tss"""
        return np.sum((self.filter_results.endog[0] - np.mean(self.filter_results.endog)) ** 2)

    @cache_readonly
    def uncentered_tss(self):
        """uncentered tss"""
        return np.sum(self.filter_results.endog[0] ** 2)

    @cache_readonly
    def ess(self):
        """ess"""
        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        """rsquared"""
        if self.k_constant:
            return 1 - self.ssr / self.centered_tss
        else:
            return 1 - self.ssr / self.uncentered_tss

    @cache_readonly
    def mse_model(self):
        """mse_model"""
        return self.ess / self.df_model

    @cache_readonly
    def mse_resid(self):
        """mse_resid"""
        return self.ssr / self.df_resid

    @cache_readonly
    def mse_total(self):
        """mse_total"""
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @Appender(MLEResults.get_prediction.__doc__)
    def get_prediction(self, start=None, end=None, dynamic=False, information_set='predicted', signal_only=False, index=None, **kwargs):
        if start is None:
            start = self.model._index[0]
        start, end, out_of_sample, prediction_index = self.model._get_prediction_index(start, end, index)
        if isinstance(dynamic, (bytes, str)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)
        if self.model._r_matrix is not None and (out_of_sample or dynamic):
            raise NotImplementedError('Cannot yet perform out-of-sample or dynamic prediction in models with constraints.')
        prediction_results = self.filter_results.predict(start, end + out_of_sample + 1, dynamic, **kwargs)
        res_obj = PredictionResults(self, prediction_results, information_set=information_set, signal_only=signal_only, row_labels=prediction_index)
        return PredictionResultsWrapper(res_obj)

    def plot_recursive_coefficient(self, variables=0, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the recursively estimated coefficients on a given variable

        Parameters
        ----------
        variables : {int, str, list[int], list[str]}, optional
            Integer index or string name of the variable whose coefficient will
            be plotted. Can also be an iterable of integers or strings. Default
            is the first variable.
        alpha : float, optional
            The confidence intervals for the coefficient are (1 - alpha) %
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        if isinstance(variables, (int, str)):
            variables = [variables]
        k_variables = len(variables)
        exog_names = self.model.exog_names
        for i in range(k_variables):
            variable = variables[i]
            if isinstance(variable, str):
                variables[i] = exog_names.index(variable)
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        for i in range(k_variables):
            variable = variables[i]
            ax = fig.add_subplot(k_variables, 1, i + 1)
            if hasattr(self.data, 'dates') and self.data.dates is not None:
                dates = self.data.dates._mpl_repr()
            else:
                dates = np.arange(self.nobs)
            d = max(self.nobs_diffuse, self.loglikelihood_burn)
            coef = self.recursive_coefficients
            ax.plot(dates[d:], coef.filtered[variable, d:], label='Recursive estimates: %s' % exog_names[variable])
            handles, labels = ax.get_legend_handles_labels()
            if alpha is not None:
                critical_value = norm.ppf(1 - alpha / 2.0)
                std_errors = np.sqrt(coef.filtered_cov[variable, variable, :])
                ci_lower = coef.filtered[variable] - critical_value * std_errors
                ci_upper = coef.filtered[variable] + critical_value * std_errors
                ci_poly = ax.fill_between(dates[d:], ci_lower[d:], ci_upper[d:], alpha=0.2)
                ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
                if i == 0:
                    p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])
                    handles.append(p)
                    labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)
            if i < k_variables - 1:
                ax.xaxis.set_ticklabels([])
        fig.tight_layout()
        return fig

    def _cusum_significance_bounds(self, alpha, ddof=0, points=None):
        """
        Parameters
        ----------
        alpha : float, optional
            The significance bound is alpha %.
        ddof : int, optional
            The number of periods additional to `k_exog` to exclude in
            constructing the bounds. Default is zero. This is usually used
            only for testing purposes.
        points : iterable, optional
            The points at which to evaluate the significance bounds. Default is
            two points, beginning and end of the sample.

        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lw, uw) because they burn the first k_exog + 1 periods instead of the
        first k_exog. If this change is performed
        (so that `tmp = (self.nobs - d - 1)**0.5`), then the output here
        matches cusum6.

        The cusum6 behavior does not seem to be consistent with
        Brown et al. (1975); it is likely they did that because they needed
        three initial observations to get the initial OLS estimates, whereas
        we do not need to do that.
        """
        if alpha == 0.01:
            scalar = 1.143
        elif alpha == 0.05:
            scalar = 0.948
        elif alpha == 0.1:
            scalar = 0.95
        else:
            raise ValueError('Invalid significance level.')
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        tmp = (self.nobs - d - ddof) ** 0.5

        def upper_line(x):
            return scalar * tmp + 2 * scalar * (x - d) / tmp
        if points is None:
            points = np.array([d, self.nobs])
        return (-upper_line(points), upper_line(points))

    def plot_cusum(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the CUSUM statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM statistic
        moves out of the significance bounds.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        ax.plot(dates[d:], self.cusum, label='CUSUM')
        ax.hlines(0, dates[d], dates[-1], color='k', alpha=0.3)
        lower_line, upper_line = self._cusum_significance_bounds(alpha)
        ax.plot([dates[d], dates[-1]], upper_line, 'k--', label='%d%% significance' % (alpha * 100))
        ax.plot([dates[d], dates[-1]], lower_line, 'k--')
        ax.legend(loc=legend_loc)
        return fig

    def _cusum_squares_significance_bounds(self, alpha, points=None):
        """
        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lww, uww) because they use a different method for computing the
        critical value; in particular, they use tabled values from
        Table C, pp. 364-365 of "The Econometric Analysis of Time Series"
        Harvey, (1990), and use the value given to 99 observations for any
        larger number of observations. In contrast, we use the approximating
        critical values suggested in Edgerton and Wells (1994) which allows
        computing relatively good approximations for any number of
        observations.
        """
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        n = 0.5 * (self.nobs - d) - 1
        try:
            ix = [0.1, 0.05, 0.025, 0.01, 0.005].index(alpha / 2)
        except ValueError:
            raise ValueError('Invalid significance level.')
        scalars = _cusum_squares_scalars[:, ix]
        crit = scalars[0] / n ** 0.5 + scalars[1] / n + scalars[2] / n ** 1.5
        if points is None:
            points = np.array([d, self.nobs])
        line = (points - d) / (self.nobs - d)
        return (line - crit, line + crit)

    def plot_cusum_squares(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the CUSUM of squares statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM of squares
        statistic moves out of the significance bounds.

        Critical values used in creating the significance bounds are computed
        using the approximate formula of [1]_.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        .. [1] Edgerton, David, and Curt Wells. 1994.
           "Critical Values for the Cusumsq Statistic
           in Medium and Large Sized Samples."
           Oxford Bulletin of Economics and Statistics 56 (3): 355-65.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        ax.plot(dates[d:], self.cusum_squares, label='CUSUM of squares')
        ref_line = (np.arange(d, self.nobs) - d) / (self.nobs - d)
        ax.plot(dates[d:], ref_line, 'k', alpha=0.3)
        lower_line, upper_line = self._cusum_squares_significance_bounds(alpha)
        ax.plot([dates[d], dates[-1]], upper_line, 'k--', label='%d%% significance' % (alpha * 100))
        ax.plot([dates[d], dates[-1]], lower_line, 'k--')
        ax.legend(loc=legend_loc)
        return fig