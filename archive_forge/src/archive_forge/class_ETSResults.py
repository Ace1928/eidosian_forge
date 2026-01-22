from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
class ETSResults(base.StateSpaceMLEResults):
    """
    Results from an error, trend, seasonal (ETS) exponential smoothing model
    """

    def __init__(self, model, params, results):
        yhat, xhat = results
        self._llf = model.loglike(params)
        self._residuals = model._residuals(yhat)
        self._fittedvalues = yhat
        scale = np.mean(self._residuals ** 2)
        super().__init__(model, params, scale=scale)
        model_definition_attrs = ['short_name', 'error', 'trend', 'seasonal', 'damped_trend', 'has_trend', 'has_seasonal', 'seasonal_periods', 'initialization_method']
        for attr in model_definition_attrs:
            setattr(self, attr, getattr(model, attr))
        self.param_names = ['%s (fixed)' % name if name in self.fixed_params else name for name in self.model.param_names or []]
        internal_params = self.model._internal_params(params)
        self.states = xhat
        if self.model.use_pandas:
            states = self.states.iloc
        else:
            states = self.states
        self.initial_state = np.zeros(model._k_initial_states)
        self.level = states[:, 0]
        self.initial_level = internal_params[4]
        self.initial_state[0] = self.initial_level
        self.alpha = self.params[0]
        self.smoothing_level = self.alpha
        if self.has_trend:
            self.slope = states[:, 1]
            self.initial_trend = internal_params[5]
            self.initial_state[1] = self.initial_trend
            self.beta = self.params[1]
            self.smoothing_trend = self.beta
        if self.has_seasonal:
            self.season = states[:, self.model._seasonal_index]
            self.initial_seasonal = internal_params[6:][::-1]
            self.initial_state[self.model._seasonal_index:] = self.initial_seasonal
            self.gamma = self.params[self.model._seasonal_index]
            self.smoothing_seasonal = self.gamma
        if self.damped_trend:
            self.phi = internal_params[3]
            self.damping_trend = self.phi
        k_free_params = self.k_params - len(self.fixed_params)
        self.df_model = k_free_params + 1
        self.mean_resid = np.mean(self.resid)
        self.scale_resid = np.std(self.resid, ddof=1)
        self.standardized_forecasts_error = (self.resid - self.mean_resid) / self.scale_resid
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = 'approx'
        self._cache = {}
        self._cov_approx_complex_step = True
        self._cov_approx_centered = False
        approx_type_str = 'complex-step'
        try:
            self._rank = None
            if self.k_params == 0:
                self.cov_params_default = np.zeros((0, 0))
                self._rank = 0
                self.cov_kwds['description'] = 'No parameters estimated.'
            else:
                self.cov_params_default = self.cov_params_approx
                self.cov_kwds['description'] = descriptions['approx'].format(approx_type=approx_type_str)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = 'Covariance matrix could not be calculated: singular. information matrix.'

    @cache_readonly
    def nobs_effective(self):
        return self.nobs

    @cache_readonly
    def fittedvalues(self):
        return self._fittedvalues

    @cache_readonly
    def resid(self):
        return self._residuals

    @cache_readonly
    def llf(self):
        """
        log-likelihood function evaluated at the fitted params
        """
        return self._llf

    def _get_prediction_params(self, start_idx):
        """
        Returns internal parameter representation of smoothing parameters and
        "initial" states for prediction/simulation, that is the states just
        before the first prediction/simulation step.
        """
        internal_params = self.model._internal_params(self.params)
        if start_idx == 0:
            return internal_params
        else:
            internal_states = self.model._get_internal_states(self.states, self.params)
            start_state = np.empty(6 + self.seasonal_periods)
            start_state[0:4] = internal_params[0:4]
            start_state[4:] = internal_states[start_idx - 1, :]
            return start_state

    def _relative_forecast_variance(self, steps):
        """
        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        h = steps
        alpha = self.smoothing_level
        if self.has_trend:
            beta = self.smoothing_trend
        if self.has_seasonal:
            gamma = self.smoothing_seasonal
            m = self.seasonal_periods
            k = np.asarray((h - 1) / m, dtype=int)
        if self.damped_trend:
            phi = self.damping_trend
        model = self.model.short_name
        if model == 'ANN':
            return 1 + alpha ** 2 * (h - 1)
        elif model == 'AAN':
            return 1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 * h / 6 * (2 * h - 1))
        elif model == 'AAdN':
            return 1 + alpha ** 2 * (h - 1) + beta * phi * h / (1 - phi) ** 2 * (2 * alpha * (1 - phi) + beta * phi) - beta * phi * (1 - phi ** h) / ((1 - phi) ** 2 * (1 - phi ** 2)) * (2 * alpha * (1 - phi ** 2) + beta * phi * (1 + 2 * phi - phi ** h))
        elif model == 'ANA':
            return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma)
        elif model == 'AAA':
            return 1 + (h - 1) * (alpha ** 2 + alpha * beta * h + beta ** 2 / 6 * h * (2 * h - 1)) + gamma * k * (2 * alpha + gamma + beta * m * (k + 1))
        elif model == 'AAdA':
            return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma) + beta * phi * h / (1 - phi) ** 2 * (2 * alpha * (1 - phi) + beta * phi) - beta * phi * (1 - phi ** h) / ((1 - phi) ** 2 * (1 - phi ** 2)) * (2 * alpha * (1 - phi ** 2) + beta * phi * (1 + 2 * phi - phi ** h)) + 2 * beta * gamma * phi / ((1 - phi) * (1 - phi ** m)) * (k * (1 - phi ** m) - phi ** m * (1 - phi ** (m * k)))
        else:
            raise NotImplementedError

    def simulate(self, nsimulations, anchor=None, repetitions=1, random_errors=None, random_state=None):
        """
        Random simulations using the state space formulation.

        Parameters
        ----------
        nsimulations : int
            The number of simulation steps.
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample (i.e. using the
            initial values as simulation anchor), and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
            Note: `anchor` corresponds to the observation right before the
            `start` observation in the `predict` method.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        random_errors : optional
            Specifies how the random errors should be obtained. Can be one of
            the following:

            * ``None``: Random normally distributed values with variance
              estimated from the fit errors drawn from numpy's standard
              RNG (can be seeded with the `random_state` argument). This is the
              default option.
            * A distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm``: Fits the distribution function to the fit
              errors and draws from the fitted distribution.
              Note the difference between ``scipy.stats.norm`` and
              ``scipy.stats.norm()``, the latter one is a frozen distribution
              function.
            * A frozen distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm(scale=2)``: Draws from the frozen distribution
              function.
            * A ``np.ndarray`` with shape (`nsimulations`, `repetitions`): Uses
              the given values as random errors.
            * ``"bootstrap"``: Samples the random errors from the fit errors.

        random_state : int or np.random.RandomState, optional
            A seed for the random number generator or a
            ``np.random.RandomState`` object. Only used if `random_errors` is
            ``None``. Default is ``None``.

        Returns
        -------
        sim : pd.Series, pd.DataFrame or np.ndarray
            An ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame`` of simulated
            values.
            If the original data was a ``pd.Series`` or ``pd.DataFrame``, `sim`
            will be a ``pd.Series`` if `repetitions` is 1, and a
            ``pd.DataFrame`` of shape (`nsimulations`, `repetitions`) else.
            Otherwise, if `repetitions` is 1, a ``np.ndarray`` of shape
            (`nsimulations`,) is returned, and if `repetitions` is not 1 a
            ``np.ndarray`` of shape (`nsimulations`, `repetitions`) is
            returned.
        """
        "\n        Implementation notes\n        --------------------\n        The simulation is based on the state space model of the Holt-Winter's\n        methods. The state space model assumes that the true value at time\n        :math:`t` is randomly distributed around the prediction value.\n        If using the additive error model, this means:\n\n        .. math::\n\n            y_t &= \\hat{y}_{t|t-1} + e_t\\\\\n            e_t &\\sim \\mathcal{N}(0, \\sigma^2)\n\n        Using the multiplicative error model:\n\n        .. math::\n\n            y_t &= \\hat{y}_{t|t-1} \\cdot (1 + e_t)\\\\\n            e_t &\\sim \\mathcal{N}(0, \\sigma^2)\n\n        Inserting these equations into the smoothing equation formulation leads\n        to the state space equations. The notation used here follows\n        [1]_.\n\n        Additionally,\n\n        .. math::\n\n           B_t = b_{t-1} \\circ_d \\phi\\\\\n           L_t = l_{t-1} \\circ_b B_t\\\\\n           S_t = s_{t-m}\\\\\n           Y_t = L_t \\circ_s S_t,\n\n        where :math:`\\circ_d` is the operation linking trend and damping\n        parameter (multiplication if the trend is additive, power if the trend\n        is multiplicative), :math:`\\circ_b` is the operation linking level and\n        trend (addition if the trend is additive, multiplication if the trend\n        is multiplicative), and :math:'\\circ_s` is the operation linking\n        seasonality to the rest.\n\n        The state space equations can then be formulated as\n\n        .. math::\n\n           y_t = Y_t + \\eta \\cdot e_t\\\\\n           l_t = L_t + \\alpha \\cdot (M_e \\cdot L_t + \\kappa_l) \\cdot e_t\\\\\n           b_t = B_t + \\beta \\cdot (M_e \\cdot B_t+\\kappa_b) \\cdot e_t\\\\\n           s_t = S_t + \\gamma \\cdot (M_e \\cdot S_t + \\kappa_s) \\cdot e_t\\\\\n\n        with\n\n        .. math::\n\n           \\eta &= \\begin{cases}\n                       Y_t\\quad\\text{if error is multiplicative}\\\\\n                       1\\quad\\text{else}\n                   \\end{cases}\\\\\n           M_e &= \\begin{cases}\n                       1\\quad\\text{if error is multiplicative}\\\\\n                       0\\quad\\text{else}\n                   \\end{cases}\\\\\n\n        and, when using the additive error model,\n\n        .. math::\n\n           \\kappa_l &= \\begin{cases}\n                       \\frac{1}{S_t}\\quad\n                       \\text{if seasonality is multiplicative}\\\\\n                       1\\quad\\text{else}\n                   \\end{cases}\\\\\n           \\kappa_b &= \\begin{cases}\n                       \\frac{\\kappa_l}{l_{t-1}}\\quad\n                       \\text{if trend is multiplicative}\\\\\n                       \\kappa_l\\quad\\text{else}\n                   \\end{cases}\\\\\n           \\kappa_s &= \\begin{cases}\n                       \\frac{1}{L_t}\\quad\n                       \\text{if seasonality is multiplicative}\\\\\n                       1\\quad\\text{else}\n                   \\end{cases}\n\n        When using the multiplicative error model\n\n        .. math::\n\n           \\kappa_l &= \\begin{cases}\n                       0\\quad\n                       \\text{if seasonality is multiplicative}\\\\\n                       S_t\\quad\\text{else}\n                   \\end{cases}\\\\\n           \\kappa_b &= \\begin{cases}\n                       \\frac{\\kappa_l}{l_{t-1}}\\quad\n                       \\text{if trend is multiplicative}\\\\\n                       \\kappa_l + l_{t-1}\\quad\\text{else}\n                   \\end{cases}\\\\\n           \\kappa_s &= \\begin{cases}\n                       0\\quad\\text{if seasonality is multiplicative}\\\\\n                       L_t\\quad\\text{else}\n                   \\end{cases}\n\n        References\n        ----------\n        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) *Forecasting:\n           principles and practice*, 2nd edition, OTexts: Melbourne,\n           Australia. OTexts.com/fpp2. Accessed on February 28th 2020.\n        "
        start_idx = self._get_prediction_start_index(anchor)
        start_params = self._get_prediction_params(start_idx)
        x = np.zeros((nsimulations, self.model._k_states_internal))
        is_fixed = np.zeros(len(start_params), dtype=np.int64)
        fixed_values = np.zeros_like(start_params)
        alpha, beta_star, gamma_star, phi, m, _ = smooth._initialize_ets_smooth(start_params, x, is_fixed, fixed_values)
        beta = alpha * beta_star
        gamma = (1 - alpha) * gamma_star
        nstates = x.shape[1]
        x = np.tile(np.reshape(x, (nsimulations, nstates, 1)), repetitions)
        y = np.empty((nsimulations, repetitions))
        sigma = np.sqrt(self.scale)
        if isinstance(random_errors, np.ndarray):
            if random_errors.shape != (nsimulations, repetitions):
                raise ValueError('If random is an ndarray, it must have shape (nsimulations, repetitions)!')
            eps = random_errors
        elif random_errors == 'bootstrap':
            eps = np.random.choice(self.resid, size=(nsimulations, repetitions), replace=True)
        elif random_errors is None:
            if random_state is None:
                eps = np.random.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
                eps = rng.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, np.random.RandomState):
                eps = random_state.randn(nsimulations, repetitions) * sigma
            else:
                raise ValueError('Argument random_state must be None, an integer, or an instance of np.random.RandomState')
        elif isinstance(random_errors, (rv_continuous, rv_discrete)):
            params = random_errors.fit(self.resid)
            eps = random_errors.rvs(*params, size=(nsimulations, repetitions))
        elif isinstance(random_errors, rv_frozen):
            eps = random_errors.rvs(size=(nsimulations, repetitions))
        else:
            raise ValueError('Argument random_errors has unexpected value!')
        mul_seasonal = self.seasonal == 'mul'
        mul_trend = self.trend == 'mul'
        mul_error = self.error == 'mul'
        if mul_trend:
            op_b = np.multiply
            op_d = np.power
        else:
            op_b = np.add
            op_d = np.multiply
        if mul_seasonal:
            op_s = np.multiply
        else:
            op_s = np.add
        for t in range(nsimulations):
            B = op_d(x[t - 1, 1, :], phi)
            L = op_b(x[t - 1, 0, :], B)
            S = x[t - 1, 2 + m - 1, :]
            Y = op_s(L, S)
            if self.error == 'add':
                eta = 1
                kappa_l = 1 / S if mul_seasonal else 1
                kappa_b = kappa_l / x[t - 1, 0, :] if mul_trend else kappa_l
                kappa_s = 1 / L if mul_seasonal else 1
            else:
                eta = Y
                kappa_l = 0 if mul_seasonal else S
                kappa_b = kappa_l / x[t - 1, 0, :] if mul_trend else kappa_l + x[t - 1, 0, :]
                kappa_s = 0 if mul_seasonal else L
            y[t, :] = Y + eta * eps[t, :]
            x[t, 0, :] = L + alpha * (mul_error * L + kappa_l) * eps[t, :]
            x[t, 1, :] = B + beta * (mul_error * B + kappa_b) * eps[t, :]
            x[t, 2, :] = S + gamma * (mul_error * S + kappa_s) * eps[t, :]
            x[t, 3:, :] = x[t - 1, 2:-1, :]
        if repetitions > 1:
            names = ['simulation.%d' % num for num in range(repetitions)]
        else:
            names = 'simulation'
        return self.model._wrap_data(y, start_idx, start_idx + nsimulations - 1, names=names)

    def forecast(self, steps=1):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        return self._forecast(steps, 'end')

    def _forecast(self, steps, anchor):
        """
        Dynamic prediction/forecasting
        """
        return self.simulate(steps, anchor=anchor, random_errors=np.zeros((steps, 1)))

    def _handle_prediction_index(self, start, dynamic, end, index):
        if start is None:
            start = 0
        start, end, out_of_sample, _ = self.model._get_prediction_index(start, end, index)
        if start > end + out_of_sample + 1:
            raise ValueError('Prediction start cannot lie outside of the sample.')
        if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)
            dynamic = dynamic - start
        elif isinstance(dynamic, bool):
            if dynamic:
                dynamic = 0
            else:
                dynamic = end + 1 - start
        if dynamic == 0:
            start_smooth = None
            end_smooth = None
            nsmooth = 0
            start_dynamic = start
        else:
            start_smooth = start
            end_smooth = min(start + dynamic - 1, end)
            nsmooth = max(end_smooth - start_smooth + 1, 0)
            start_dynamic = start + dynamic
        if start_dynamic == 0:
            anchor_dynamic = 'start'
        else:
            anchor_dynamic = start_dynamic - 1
        end_dynamic = end + out_of_sample
        ndynamic = end_dynamic - start_dynamic + 1
        return (start, end, start_smooth, end_smooth, anchor_dynamic, start_dynamic, end_dynamic, nsmooth, ndynamic, index)

    def predict(self, start=None, end=None, dynamic=False, index=None):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.

        Returns
        -------
        forecast : array_like or pd.Series.
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict,) array. If original data was a pd.Series
            or DataFrame, a pd.Series is returned.
        """
        start, end, start_smooth, end_smooth, anchor_dynamic, _, end_dynamic, nsmooth, ndynamic, index = self._handle_prediction_index(start, dynamic, end, index)
        y = np.empty(nsmooth + ndynamic)
        if nsmooth > 0:
            y[0:nsmooth] = self.fittedvalues[start_smooth:end_smooth + 1]
        if ndynamic > 0:
            y[nsmooth:] = self._forecast(ndynamic, anchor_dynamic)
        if start > end + 1:
            ndiscard = start - (end + 1)
            y = y[ndiscard:]
        return self.model._wrap_data(y, start, end_dynamic)

    def get_prediction(self, start=None, end=None, dynamic=False, index=None, method=None, simulate_repetitions=1000, **simulate_kwargs):
        """
        Calculates mean prediction and prediction intervals.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        method : str or None, optional
            Method to use for calculating prediction intervals. 'exact'
            (default, if available) or 'simulated'.
        simulate_repetitions : int, optional
            Number of simulation repetitions for calculating prediction
            intervals when ``method='simulated'``. Default is 1000.
        **simulate_kwargs :
            Additional arguments passed to the ``simulate`` method.

        Returns
        -------
        PredictionResults
            Predicted mean values and prediction intervals
        """
        return PredictionResultsWrapper(PredictionResults(self, start, end, dynamic, index, method, simulate_repetitions, **simulate_kwargs))

    def summary(self, alpha=0.05, start=None):
        """
        Summarize the fitted model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        model_name = f'ETS({self.short_name})'
        summary = super().summary(alpha=alpha, start=start, title='ETS Results', model_name=model_name)
        if self.model.initialization_method != 'estimated':
            params = np.array(self.initial_state)
            if params.ndim > 1:
                params = params[0]
            names = self.model.initial_state_names
            param_header = ['initialization method: %s' % self.model.initialization_method]
            params_stubs = names
            params_data = [[forg(params[i], prec=4)] for i in range(len(params))]
            initial_state_table = SimpleTable(params_data, param_header, params_stubs, txt_fmt=fmt_params)
            summary.tables.insert(-1, initial_state_table)
        return summary