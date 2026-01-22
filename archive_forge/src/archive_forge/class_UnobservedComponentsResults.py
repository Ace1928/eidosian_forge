from warnings import warn
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
class UnobservedComponentsResults(MLEResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        self.df_resid = np.inf
        self._init_kwds = self.model._get_init_kwds()
        self._k_states_by_type = {'seasonal': self.model._k_seasonal_states, 'freq_seasonal': self.model._k_freq_seas_states, 'cycle': self.model._k_cycle_states}
        self.specification = Bunch(**{'level': self.model.level, 'trend': self.model.trend, 'seasonal_periods': self.model.seasonal_periods, 'seasonal': self.model.seasonal, 'freq_seasonal': self.model.freq_seasonal, 'freq_seasonal_periods': self.model.freq_seasonal_periods, 'freq_seasonal_harmonics': self.model.freq_seasonal_harmonics, 'cycle': self.model.cycle, 'ar_order': self.model.ar_order, 'autoregressive': self.model.autoregressive, 'irregular': self.model.irregular, 'stochastic_level': self.model.stochastic_level, 'stochastic_trend': self.model.stochastic_trend, 'stochastic_seasonal': self.model.stochastic_seasonal, 'stochastic_freq_seasonal': self.model.stochastic_freq_seasonal, 'stochastic_cycle': self.model.stochastic_cycle, 'damped_cycle': self.model.damped_cycle, 'regression': self.model.regression, 'mle_regression': self.model.mle_regression, 'k_exog': self.model.k_exog, 'trend_specification': self.model.trend_specification})

    @property
    def level(self):
        """
        Estimates of unobserved level component

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
        if spec.level:
            offset = 0
            out = Bunch(filtered=self.filtered_state[offset], filtered_cov=self.filtered_state_cov[offset, offset], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def trend(self):
        """
        Estimates of of unobserved trend component

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
        if spec.trend:
            offset = int(spec.level)
            out = Bunch(filtered=self.filtered_state[offset], filtered_cov=self.filtered_state_cov[offset, offset], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def seasonal(self):
        """
        Estimates of unobserved seasonal component

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
        if spec.seasonal:
            offset = int(spec.trend + spec.level)
            out = Bunch(filtered=self.filtered_state[offset], filtered_cov=self.filtered_state_cov[offset, offset], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def freq_seasonal(self):
        """
        Estimates of unobserved frequency domain seasonal component(s)

        Returns
        -------
        out: list of Bunch instances
            Each item has the following attributes:

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
        out = []
        spec = self.specification
        if spec.freq_seasonal:
            previous_states_offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'])
            previous_f_seas_offset = 0
            for ix, h in enumerate(spec.freq_seasonal_harmonics):
                offset = previous_states_offset + previous_f_seas_offset
                period = spec.freq_seasonal_periods[ix]
                states_in_sum = np.arange(0, 2 * h, 2)
                filtered_state = np.sum([self.filtered_state[offset + j] for j in states_in_sum], axis=0)
                filtered_cov = np.sum([self.filtered_state_cov[offset + j, offset + j] for j in states_in_sum], axis=0)
                item = Bunch(filtered=filtered_state, filtered_cov=filtered_cov, smoothed=None, smoothed_cov=None, offset=offset, pretty_name='seasonal {p}({h})'.format(p=repr(period), h=repr(h)))
                if self.smoothed_state is not None:
                    item.smoothed = np.sum([self.smoothed_state[offset + j] for j in states_in_sum], axis=0)
                if self.smoothed_state_cov is not None:
                    item.smoothed_cov = np.sum([self.smoothed_state_cov[offset + j, offset + j] for j in states_in_sum], axis=0)
                out.append(item)
                previous_f_seas_offset += 2 * h
        return out

    @property
    def cycle(self):
        """
        Estimates of unobserved cycle component

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
        if spec.cycle:
            offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'] + self._k_states_by_type['freq_seasonal'])
            out = Bunch(filtered=self.filtered_state[offset], filtered_cov=self.filtered_state_cov[offset, offset], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def autoregressive(self):
        """
        Estimates of unobserved autoregressive component

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
        if spec.autoregressive:
            offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'] + self._k_states_by_type['freq_seasonal'] + self._k_states_by_type['cycle'])
            out = Bunch(filtered=self.filtered_state[offset], filtered_cov=self.filtered_state_cov[offset, offset], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def regression_coefficients(self):
        """
        Estimates of unobserved regression coefficients

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
        if spec.regression:
            if spec.mle_regression:
                import warnings
                warnings.warn('Regression coefficients estimated via maximum likelihood. Estimated coefficients are available in the parameters list, not as part of the state vector.', OutputWarning)
            else:
                offset = int(spec.trend + spec.level + self._k_states_by_type['seasonal'] + self._k_states_by_type['freq_seasonal'] + self._k_states_by_type['cycle'] + spec.ar_order)
                start = offset
                end = offset + spec.k_exog
                out = Bunch(filtered=self.filtered_state[start:end], filtered_cov=self.filtered_state_cov[start:end, start:end], smoothed=None, smoothed_cov=None, offset=offset)
                if self.smoothed_state is not None:
                    out.smoothed = self.smoothed_state[start:end]
                if self.smoothed_state_cov is not None:
                    out.smoothed_cov = self.smoothed_state_cov[start:end, start:end]
        return out

    def plot_components(self, which=None, alpha=0.05, observed=True, level=True, trend=True, seasonal=True, freq_seasonal=True, cycle=True, autoregressive=True, legend_loc='upper right', fig=None, figsize=None):
        """
        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered', 'smoothed'}, or None, optional
            Type of state estimate to plot. Default is 'smoothed' if smoothed
            results are available otherwise 'filtered'.
        alpha : float, optional
            The confidence intervals for the components are (1 - alpha) %
        observed : bool, optional
            Whether or not to plot the observed series against
            one-step-ahead predictions.
            Default is True.
        level : bool, optional
            Whether or not to plot the level component, if applicable.
            Default is True.
        trend : bool, optional
            Whether or not to plot the trend component, if applicable.
            Default is True.
        seasonal : bool, optional
            Whether or not to plot the seasonal component, if applicable.
            Default is True.
        freq_seasonal : bool, optional
            Whether or not to plot the frequency domain seasonal component(s),
            if applicable. Default is True.
        cycle : bool, optional
            Whether or not to plot the cyclical component, if applicable.
            Default is True.
        autoregressive : bool, optional
            Whether or not to plot the autoregressive state, if applicable.
            Default is True.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        If all options are included in the model and selected, this produces
        a 6x1 plot grid with the following plots (ordered top-to-bottom):

        0. Observed series against predicted series
        1. Level
        2. Trend
        3. Seasonal
        4. Freq Seasonal
        5. Cycle
        6. Autoregressive

        Specific subplots will be removed if the component is not present in
        the estimated model or if the corresponding keyword argument is set to
        False.

        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'
        spec = self.specification
        comp = [('level', level and spec.level), ('trend', trend and spec.trend), ('seasonal', seasonal and spec.seasonal)]
        if freq_seasonal and spec.freq_seasonal:
            for ix, _ in enumerate(spec.freq_seasonal_periods):
                key = f'freq_seasonal_{ix!r}'
                comp.append((key, True))
        comp.extend([('cycle', cycle and spec.cycle), ('autoregressive', autoregressive and spec.autoregressive)])
        components = dict(comp)
        llb = self.filter_results.loglikelihood_burn
        k_plots = observed + np.sum(list(components.values()))
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(len(self.data.endog))
        critical_value = norm.ppf(1 - alpha / 2.0)
        plot_idx = 1
        if observed:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1
            ax.plot(dates[llb:], self.model.endog[llb:], color='k', label='Observed')
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors
            ax.plot(dates[llb:], predict[llb:], label='One-step-ahead predictions')
            ci_poly = ax.fill_between(dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2)
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)
            ax.set_title('Predicted vs observed')
        for component, is_plotted in components.items():
            if not is_plotted:
                continue
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1
            try:
                component_bunch = getattr(self, component)
                title = component.title()
            except AttributeError:
                if component.startswith('freq_seasonal_'):
                    ix = int(component.replace('freq_seasonal_', ''))
                    big_bunch = getattr(self, 'freq_seasonal')
                    component_bunch = big_bunch[ix]
                    title = component_bunch.pretty_name
                else:
                    raise
            if which not in component_bunch:
                raise ValueError('Invalid type of state estimate.')
            which_cov = '%s_cov' % which
            value = component_bunch[which]
            state_label = '{} ({})'.format(title, which)
            ax.plot(dates[llb:], value[llb:], label=state_label)
            if which_cov in component_bunch:
                std_errors = np.sqrt(component_bunch['%s_cov' % which])
                ci_lower = value - critical_value * std_errors
                ci_upper = value + critical_value * std_errors
                ci_poly = ax.fill_between(dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2)
                ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
            ax.legend(loc=legend_loc)
            ax.set_title('%s component' % title)
        if llb > 0:
            text = 'Note: The first %d observations are not shown, due to approximate diffuse initialization.'
            fig.text(0.1, 0.01, text % llb, fontsize='large')
        return fig

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=0.05, start=None):
        model_name = [self.specification.trend_specification]
        if self.specification.seasonal:
            seasonal_name = 'seasonal(%d)' % self.specification.seasonal_periods
            if self.specification.stochastic_seasonal:
                seasonal_name = 'stochastic ' + seasonal_name
            model_name.append(seasonal_name)
        if self.specification.freq_seasonal:
            for ix, is_stochastic in enumerate(self.specification.stochastic_freq_seasonal):
                periodicity = self.specification.freq_seasonal_periods[ix]
                harmonics = self.specification.freq_seasonal_harmonics[ix]
                freq_seasonal_name = 'freq_seasonal({p}({h}))'.format(p=repr(periodicity), h=repr(harmonics))
                if is_stochastic:
                    freq_seasonal_name = 'stochastic ' + freq_seasonal_name
                model_name.append(freq_seasonal_name)
        if self.specification.cycle:
            cycle_name = 'cycle'
            if self.specification.stochastic_cycle:
                cycle_name = 'stochastic ' + cycle_name
            if self.specification.damped_cycle:
                cycle_name = 'damped ' + cycle_name
            model_name.append(cycle_name)
        if self.specification.autoregressive:
            autoregressive_name = 'AR(%d)' % self.specification.ar_order
            model_name.append(autoregressive_name)
        return super().summary(alpha=alpha, start=start, title='Unobserved Components Results', model_name=model_name)