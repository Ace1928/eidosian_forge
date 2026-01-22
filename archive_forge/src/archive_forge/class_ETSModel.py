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
class ETSModel(base.StateSpaceMLEModel):
    """
    ETS models.

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    error : str, optional
        The error model. "add" (default) or "mul".
    trend : str or None, optional
        The trend component model. "add", "mul", or None (default).
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is
        False.
    seasonal : str, optional
        The seasonality model. "add", "mul", or None (default).
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Required if
        `seasonal` is not None.
    initialization_method : str, optional
        Method for initialization of the state space model. One of:

        * 'estimated' (default)
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_trend` and `initial_seasonal` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.  Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal_periods`.
        Only used if initialization is 'known'.
    bounds : dict or None, optional
        A dictionary with parameter names as keys and the respective bounds
        intervals as values (lists/tuples/arrays).
        The available parameter names are, depending on the model and
        initialization method:

        * "smoothing_level"
        * "smoothing_trend"
        * "smoothing_seasonal"
        * "damping_trend"
        * "initial_level"
        * "initial_trend"
        * "initial_seasonal.0", ..., "initial_seasonal.<m-1>"

        The default option is ``None``, in which case the traditional
        (nonlinear) bounds as described in [1]_ are used.

    Notes
    -----
    The ETS models are a family of time series models. They can be seen as a
    generalization of simple exponential smoothing to time series that contain
    trends and seasonalities. Additionally, they have an underlying state
    space model.

    An ETS model is specified by an error type (E; additive or multiplicative),
    a trend type (T; additive or multiplicative, both damped or undamped, or
    none), and a seasonality type (S; additive or multiplicative or none).
    The following gives a very short summary, a more thorough introduction can
    be found in [1]_.

    Denote with :math:`\\circ_b` the trend operation (addition or
    multiplication), with :math:`\\circ_d` the operation linking trend and
    dampening factor :math:`\\phi` (multiplication if trend is additive, power
    if trend is multiplicative), and with :math:`\\circ_s` the seasonality
    operation (addition or multiplication). Furthermore, let :math:`\\ominus`
    be the respective inverse operation (subtraction or division).

    With this, it is possible to formulate the ETS models as a forecast
    equation and 3 smoothing equations. The former is used to forecast
    observations, the latter are used to update the internal state.

    .. math::

        \\hat{y}_{t|t-1} &= (l_{t-1} \\circ_b (b_{t-1}\\circ_d \\phi))
                           \\circ_s s_{t-m}\\\\
        l_{t} &= \\alpha (y_{t} \\ominus_s s_{t-m})
                 + (1 - \\alpha) (l_{t-1} \\circ_b (b_{t-1} \\circ_d \\phi))\\\\
        b_{t} &= \\beta/\\alpha (l_{t} \\ominus_b l_{t-1})
                 + (1 - \\beta/\\alpha) b_{t-1}\\\\
        s_{t} &= \\gamma (y_t \\ominus_s (l_{t-1} \\circ_b (b_{t-1}\\circ_d\\phi))
                 + (1 - \\gamma) s_{t-m}

    The notation here follows [1]_; :math:`l_t` denotes the level at time
    :math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m`
    is the number of seasonal periods, and :math:`\\phi` a trend damping
    factor. The parameters :math:`\\alpha, \\beta, \\gamma` are the smoothing
    parameters, which are called ``smoothing_level``, ``smoothing_trend``, and
    ``smoothing_seasonal``, respectively.

    Note that the formulation above as forecast and smoothing equation does
    not distinguish different error models -- it is the same for additive and
    multiplicative errors. But the different error models lead to different
    likelihood models, and therefore will lead to different fit results.

    The error models specify how the true values :math:`y_t` are
    updated. In the additive error model,

    .. math::

        y_t = \\hat{y}_{t|t-1} + e_t,

    in the multiplicative error model,

    .. math::

        y_t = \\hat{y}_{t|t-1}\\cdot (1 + e_t).

    Using these error models, it is possible to formulate state space
    equations for the ETS models:

    .. math::

       y_t &= Y_t + \\eta \\cdot e_t\\\\
       l_t &= L_t + \\alpha \\cdot (M_e \\cdot L_t + \\kappa_l) \\cdot e_t\\\\
       b_t &= B_t + \\beta \\cdot (M_e \\cdot B_t + \\kappa_b) \\cdot e_t\\\\
       s_t &= S_t + \\gamma \\cdot (M_e \\cdot S_t+\\kappa_s)\\cdot e_t\\\\

    with

    .. math::

       B_t &= b_{t-1} \\circ_d \\phi\\\\
       L_t &= l_{t-1} \\circ_b B_t\\\\
       S_t &= s_{t-m}\\\\
       Y_t &= L_t \\circ_s S_t,

    and

    .. math::

       \\eta &= \\begin{cases}
                   Y_t\\quad\\text{if error is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}\\\\
       M_e &= \\begin{cases}
                   1\\quad\\text{if error is multiplicative}\\\\
                   0\\quad\\text{else}
               \\end{cases}\\\\

    and, when using the additive error model,

    .. math::

       \\kappa_l &= \\begin{cases}
                   \\frac{1}{S_t}\\quad
                   \\text{if seasonality is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_b &= \\begin{cases}
                   \\frac{\\kappa_l}{l_{t-1}}\\quad
                   \\text{if trend is multiplicative}\\\\
                   \\kappa_l\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_s &= \\begin{cases}
                   \\frac{1}{L_t}\\quad\\text{if seasonality is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}

    When using the multiplicative error model

    .. math::

       \\kappa_l &= \\begin{cases}
                   0\\quad
                   \\text{if seasonality is multiplicative}\\\\
                   S_t\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_b &= \\begin{cases}
                   \\frac{\\kappa_l}{l_{t-1}}\\quad
                   \\text{if trend is multiplicative}\\\\
                   \\kappa_l + l_{t-1}\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_s &= \\begin{cases}
                   0\\quad\\text{if seasonality is multiplicative}\\\\
                   L_t\\quad\\text{else}
               \\end{cases}

    When fitting an ETS model, the parameters :math:`\\alpha, \\beta`, \\gamma,
    \\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \\ldots, s_{-m}` are
    selected as maximizers of log likelihood.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
       principles and practice*, 3rd edition, OTexts: Melbourne,
       Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
    """

    def __init__(self, endog, error='add', trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, bounds=None, dates=None, freq=None, missing='none'):
        super().__init__(endog, exog=None, dates=dates, freq=freq, missing=missing)
        options = ('add', 'mul', 'additive', 'multiplicative')
        self.error = string_like(error, 'error', options=options)[:3]
        self.trend = string_like(trend, 'trend', options=options, optional=True)
        if self.trend is not None:
            self.trend = self.trend[:3]
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        self.seasonal = string_like(seasonal, 'seasonal', options=options, optional=True)
        if self.seasonal is not None:
            self.seasonal = self.seasonal[:3]
        self.has_trend = self.trend is not None
        self.has_seasonal = self.seasonal is not None
        if self.has_seasonal:
            self.seasonal_periods = int_like(seasonal_periods, 'seasonal_periods', optional=True)
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError('seasonal_periods must be larger than 1.')
        else:
            self.seasonal_periods = 1
        if np.any(self.endog <= 0) and (self.error == 'mul' or self.trend == 'mul' or self.seasonal == 'mul'):
            raise ValueError('endog must be strictly positive when using multiplicative error, trend or seasonal components.')
        if self.damped_trend and (not self.has_trend):
            raise ValueError('Can only dampen the trend component')
        self.set_initialization_method(initialization_method, initial_level, initial_trend, initial_seasonal)
        self.set_bounds(bounds)
        if self.trend == 'add' or self.trend is None:
            if self.seasonal == 'add' or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_add_add
            else:
                self._smoothing_func = smooth._ets_smooth_add_mul
        elif self.seasonal == 'add' or self.seasonal is None:
            self._smoothing_func = smooth._ets_smooth_mul_add
        else:
            self._smoothing_func = smooth._ets_smooth_mul_mul

    def set_initialization_method(self, initialization_method, initial_level=None, initial_trend=None, initial_seasonal=None):
        """
        Sets a new initialization method for the state space model.

        Parameters
        ----------
        initialization_method : str, optional
            Method for initialization of the state space model. One of:

            * 'estimated' (default)
            * 'heuristic'
            * 'known'

            If 'known' initialization is used, then `initial_level` must be
            passed, as well as `initial_trend` and `initial_seasonal` if
            applicable.
            'heuristic' uses a heuristic based on the data to estimate initial
            level, trend, and seasonal state. 'estimated' uses the same
            heuristic as initial guesses, but then estimates the initial states
            as part of the fitting process. Default is 'estimated'.
        initial_level : float, optional
            The initial level component. Only used if initialization is
            'known'.
        initial_trend : float, optional
            The initial trend component. Only used if initialization is
            'known'.
        initial_seasonal : array_like, optional
            The initial seasonal component. An array of length
            `seasonal_periods`. Only used if initialization is 'known'.
        """
        self.initialization_method = string_like(initialization_method, 'initialization_method', options=('estimated', 'known', 'heuristic'))
        if self.initialization_method == 'known':
            if initial_level is None:
                raise ValueError('`initial_level` argument must be provided when initialization method is set to "known".')
            if self.has_trend and initial_trend is None:
                raise ValueError('`initial_trend` argument must be provided for models with a trend component when initialization method is set to "known".')
            if self.has_seasonal and initial_seasonal is None:
                raise ValueError('`initial_seasonal` argument must be provided for models with a seasonal component when initialization method is set to "known".')
        elif self.initialization_method == 'heuristic':
            initial_level, initial_trend, initial_seasonal = _initialization_heuristic(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        elif self.initialization_method == 'estimated':
            if self.nobs < 10 + 2 * (self.seasonal_periods // 2):
                initial_level, initial_trend, initial_seasonal = _initialization_simple(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
            else:
                initial_level, initial_trend, initial_seasonal = _initialization_heuristic(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        if not self.has_trend:
            initial_trend = 0
        if not self.has_seasonal:
            initial_seasonal = 0
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self._internal_params_index = OrderedDict(zip(self._internal_param_names, np.arange(self._k_params_internal)))
        self._params_index = OrderedDict(zip(self.param_names, np.arange(self.k_params)))

    def set_bounds(self, bounds):
        """
        Set bounds for parameter estimation.

        Parameters
        ----------
        bounds : dict or None, optional
            A dictionary with parameter names as keys and the respective bounds
            intervals as values (lists/tuples/arrays).
            The available parameter names are in ``self.param_names``.
            The default option is ``None``, in which case the traditional
            (nonlinear) bounds as described in [1]_ are used.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        if bounds is None:
            self.bounds = {}
        else:
            if not isinstance(bounds, (dict, OrderedDict)):
                raise ValueError('bounds must be a dictionary')
            for key in bounds:
                if key not in self.param_names:
                    raise ValueError(f'Invalid key: {key} in bounds dictionary')
                bounds[key] = array_like(bounds[key], f'bounds[{key}]', shape=(2,))
            self.bounds = bounds

    @staticmethod
    def prepare_data(data):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(data.orig_endog, order='C')
        if endog.ndim != 1:
            raise ValueError('endog must be 1-dimensional')
        if endog.dtype != np.double:
            endog = np.asarray(data.orig_endog, order='C', dtype=float)
        return (endog, None)

    @property
    def nobs_effective(self):
        return self.nobs

    @property
    def k_endog(self):
        return 1

    @property
    def short_name(self):
        name = ''.join([str(s)[0].upper() for s in [self.error, self.trend, self.seasonal]])
        if self.damped_trend:
            name = name[0:2] + 'd' + name[2]
        return name

    @property
    def _param_names(self):
        param_names = ['smoothing_level']
        if self.has_trend:
            param_names += ['smoothing_trend']
        if self.has_seasonal:
            param_names += ['smoothing_seasonal']
        if self.damped_trend:
            param_names += ['damping_trend']
        if self.initialization_method == 'estimated':
            param_names += ['initial_level']
            if self.has_trend:
                param_names += ['initial_trend']
            if self.has_seasonal:
                param_names += [f'initial_seasonal.{i}' for i in range(self.seasonal_periods)]
        return param_names

    @property
    def state_names(self):
        names = ['level']
        if self.has_trend:
            names += ['trend']
        if self.has_seasonal:
            names += ['seasonal']
        return names

    @property
    def initial_state_names(self):
        names = ['initial_level']
        if self.has_trend:
            names += ['initial_trend']
        if self.has_seasonal:
            names += [f'initial_seasonal.{i}' for i in range(self.seasonal_periods)]
        return names

    @property
    def _smoothing_param_names(self):
        return ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'damping_trend']

    @property
    def _internal_initial_state_names(self):
        param_names = ['initial_level', 'initial_trend']
        param_names += [f'initial_seasonal.{i}' for i in range(self.seasonal_periods)]
        return param_names

    @property
    def _internal_param_names(self):
        return self._smoothing_param_names + self._internal_initial_state_names

    @property
    def _k_states(self):
        return 1 + int(self.has_trend) + int(self.has_seasonal)

    @property
    def _k_states_internal(self):
        return 2 + self.seasonal_periods

    @property
    def _k_smoothing_params(self):
        return self._k_states + int(self.damped_trend)

    @property
    def _k_initial_states(self):
        return 1 + int(self.has_trend) + +int(self.has_seasonal) * self.seasonal_periods

    @property
    def k_params(self):
        k = self._k_smoothing_params
        if self.initialization_method == 'estimated':
            k += self._k_initial_states
        return k

    @property
    def _k_params_internal(self):
        return 4 + 2 + self.seasonal_periods

    def _internal_params(self, params):
        """
        Converts a parameter array passed from outside to the internally used
        full parameter array.
        """
        internal = np.zeros(self._k_params_internal, dtype=params.dtype)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            internal[internal_idx] = params[i]
        if not self.damped_trend:
            internal[3] = 1
        if self.initialization_method != 'estimated':
            internal[4] = self.initial_level
            internal[5] = self.initial_trend
            if np.isscalar(self.initial_seasonal):
                internal[6:] = self.initial_seasonal
            else:
                internal[6:] = self.initial_seasonal[::-1]
        return internal

    def _model_params(self, internal):
        """
        Converts internal parameters to model parameters
        """
        params = np.empty(self.k_params)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            params[i] = internal[internal_idx]
        return params

    @property
    def _seasonal_index(self):
        return 1 + int(self.has_trend)

    def _get_states(self, xhat):
        states = np.empty((self.nobs, self._k_states))
        all_names = ['level', 'trend', 'seasonal']
        for i, name in enumerate(self.state_names):
            idx = all_names.index(name)
            states[:, i] = xhat[:, idx]
        return states

    def _get_internal_states(self, states, params):
        """
        Converts a state matrix/dataframe to the (nobs, 2+m) matrix used
        internally
        """
        internal_params = self._internal_params(params)
        if isinstance(states, (pd.Series, pd.DataFrame)):
            states = states.values
        internal_states = np.zeros((self.nobs, 2 + self.seasonal_periods))
        internal_states[:, 0] = states[:, 0]
        if self.has_trend:
            internal_states[:, 1] = states[:, 1]
        if self.has_seasonal:
            for j in range(self.seasonal_periods):
                internal_states[j:, 2 + j] = states[0:self.nobs - j, self._seasonal_index]
                internal_states[0:j, 2 + j] = internal_params[6:6 + j][::-1]
        return internal_states

    @property
    def _default_start_params(self):
        return {'smoothing_level': 0.1, 'smoothing_trend': 0.01, 'smoothing_seasonal': 0.01, 'damping_trend': 0.98}

    @property
    def _start_params(self):
        """
        Default start params in the format of external parameters.
        This should not be called directly, but by calling
        ``self.start_params``.
        """
        params = []
        for p in self._smoothing_param_names:
            if p in self.param_names:
                params.append(self._default_start_params[p])
        if self.initialization_method == 'estimated':
            lvl_idx = len(params)
            params += [self.initial_level]
            if self.has_trend:
                params += [self.initial_trend]
            if self.has_seasonal:
                initial_seasonal = self.initial_seasonal
                if self.seasonal == 'mul':
                    params[lvl_idx] *= initial_seasonal[-1]
                    initial_seasonal /= initial_seasonal[-1]
                else:
                    params[lvl_idx] += initial_seasonal[-1]
                    initial_seasonal -= initial_seasonal[-1]
                params += initial_seasonal.tolist()
        return np.array(params)

    def _convert_and_bound_start_params(self, params):
        """
        This converts start params to internal params, sets internal-only
        parameters as bounded, sets bounds for fixed parameters, and then makes
        sure that all start parameters are within the specified bounds.
        """
        internal_params = self._internal_params(params)
        for p in self._internal_param_names:
            idx = self._internal_params_index[p]
            if p not in self.param_names:
                self.bounds[p] = [internal_params[idx]] * 2
            elif self._has_fixed_params and p in self._fixed_params:
                self.bounds[p] = [self._fixed_params[p]] * 2
            if p in self.bounds:
                internal_params[idx] = np.clip(internal_params[idx] + 0.001, *self.bounds[p])
        return internal_params

    def _setup_bounds(self):
        lb = np.zeros(self._k_params_internal) + 0.0001
        ub = np.ones(self._k_params_internal) - 0.0001
        lb[3], ub[3] = (0.8, 0.98)
        if self.initialization_method == 'estimated':
            lb[4:-1] = -np.inf
            ub[4:-1] = np.inf
            if self.seasonal == 'mul':
                lb[-1], ub[-1] = (1, 1)
            else:
                lb[-1], ub[-1] = (0, 0)
        for p in self._internal_param_names:
            idx = self._internal_params_index[p]
            if p in self.bounds:
                lb[idx], ub[idx] = self.bounds[p]
        return [(lb[i], ub[i]) for i in range(self._k_params_internal)]

    def fit(self, start_params=None, maxiter=1000, full_output=True, disp=True, callback=None, return_params=False, **kwargs):
        """
        Fit an ETS model by maximizing log-likelihood.

        Log-likelihood is a function of the model parameters :math:`\\alpha,
        \\beta, \\gamma, \\phi` (depending on the chosen model), and, if
        `initialization_method` was set to `'estimated'` in the constructor,
        also the initial states :math:`l_{-1}, b_{-1}, s_{-1}, \\ldots, s_{-m}`.

        The fit is performed using the L-BFGS algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial values for parameters that will be optimized. If this is
            ``None``, default values will be used.
            The length of this depends on the chosen model. This should contain
            the parameters in the following order, skipping parameters that do
            not exist in the chosen model.

            * `smoothing_level` (:math:`\\alpha`)
            * `smoothing_trend` (:math:`\\beta`)
            * `smoothing_seasonal` (:math:`\\gamma`)
            * `damping_trend` (:math:`\\phi`)

            If ``initialization_method`` was set to ``'estimated'`` (the
            default), additionally, the parameters

            * `initial_level` (:math:`l_{-1}`)
            * `initial_trend` (:math:`l_{-1}`)
            * `initial_seasonal.0` (:math:`s_{-1}`)
            * ...
            * `initial_seasonal.<m-1>` (:math:`s_{-m}`)

            also have to be specified.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results : ETSResults
        """
        if start_params is None:
            start_params = self.start_params
        else:
            start_params = np.asarray(start_params)
        if self._has_fixed_params and len(self._free_params_index) == 0:
            final_params = np.asarray(list(self._fixed_params.values()))
            mlefit = Bunch(params=start_params, mle_retvals=None, mle_settings=None)
        else:
            internal_start_params = self._convert_and_bound_start_params(start_params)
            bounds = self._setup_bounds()
            use_beta_star = 'smoothing_trend' not in self.bounds
            if use_beta_star:
                internal_start_params[1] /= internal_start_params[0]
            use_gamma_star = 'smoothing_seasonal' not in self.bounds
            if use_gamma_star:
                internal_start_params[2] /= 1 - internal_start_params[0]
            is_fixed = np.zeros(self._k_params_internal, dtype=np.int64)
            fixed_values = np.empty_like(internal_start_params)
            params_without_fixed = []
            kwargs['bounds'] = []
            for i in range(self._k_params_internal):
                if bounds[i][0] == bounds[i][1]:
                    is_fixed[i] = True
                    fixed_values[i] = bounds[i][0]
                else:
                    params_without_fixed.append(internal_start_params[i])
                    kwargs['bounds'].append(bounds[i])
            params_without_fixed = np.asarray(params_without_fixed)
            yhat = np.zeros(self.nobs)
            xhat = np.zeros((self.nobs, self._k_states_internal))
            kwargs['approx_grad'] = True
            with self.use_internal_loglike():
                mlefit = super().fit(params_without_fixed, fargs=(yhat, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star), method='lbfgs', maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, skip_hessian=True, **kwargs)
            fitted_params = np.empty_like(internal_start_params)
            idx_without_fixed = 0
            for i in range(self._k_params_internal):
                if is_fixed[i]:
                    fitted_params[i] = fixed_values[i]
                else:
                    fitted_params[i] = mlefit.params[idx_without_fixed]
                    idx_without_fixed += 1
            if use_beta_star:
                fitted_params[1] *= fitted_params[0]
            if use_gamma_star:
                fitted_params[2] *= 1 - fitted_params[0]
            final_params = self._model_params(fitted_params)
        if return_params:
            return final_params
        else:
            result = self.smooth(final_params)
            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings
            return result

    def _loglike_internal(self, params, yhat, xhat, is_fixed=None, fixed_values=None, use_beta_star=False, use_gamma_star=False):
        """
        Log-likelihood function to be called from fit to avoid reallocation of
        memory.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m]). If there are no fixed values this must
            be in the format of internal parameters. Otherwise the fixed values
            are skipped.
        yhat : np.ndarray
            Array of size (n,) where fitted values will be written to.
        xhat : np.ndarray
            Array of size (n, _k_states_internal) where fitted states will be
            written to.
        is_fixed : np.ndarray or None
            Boolean array indicating values which are fixed during fitting.
            This must have the full length of internal parameters.
        fixed_values : np.ndarray or None
            Array of fixed values (arbitrary values for non-fixed parameters)
            This must have the full length of internal parameters.
        use_beta_star : boolean
            Whether to internally use beta_star as parameter
        use_gamma_star : boolean
            Whether to internally use gamma_star as parameter
        """
        if np.iscomplexobj(params):
            data = np.asarray(self.endog, dtype=complex)
        else:
            data = self.endog
        if is_fixed is None:
            is_fixed = np.zeros(self._k_params_internal, dtype=np.int64)
            fixed_values = np.empty(self._k_params_internal, dtype=params.dtype)
        else:
            is_fixed = np.ascontiguousarray(is_fixed, dtype=np.int64)
        self._smoothing_func(params, data, yhat, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star)
        res = self._residuals(yhat, data=data)
        logL = -self.nobs / 2 * (np.log(2 * np.pi * np.mean(res ** 2)) + 1)
        if self.error == 'mul':
            yhat[yhat <= 0] = 1 / (1e-08 * (1 + np.abs(yhat[yhat <= 0])))
            logL -= np.sum(np.log(yhat))
        return logL

    @contextlib.contextmanager
    def use_internal_loglike(self):
        external_loglike = self.loglike
        self.loglike = self._loglike_internal
        try:
            yield
        finally:
            self.loglike = external_loglike

    def loglike(self, params, **kwargs):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m])

        Notes
        -----
        The log-likelihood of a exponential smoothing model is [1]_:

        .. math::

           l(\\theta, x_0|y) = - \\frac{n}{2}(\\log(2\\pi s^2) + 1)
                              - \\sum\\limits_{t=1}^n \\log(k_t)

        with

        .. math::

           s^2 = \\frac{1}{n}\\sum\\limits_{t=1}^n \\frac{(\\hat{y}_t - y_t)^2}{k_t}

        where :math:`k_t = 1` for the additive error model and :math:`k_t =
        y_t` for the multiplicative error model.

        References
        ----------
        .. [1] J. K. Ord, A. B. Koehler R. D. and Snyder (1997). Estimation and
           Prediction for a Class of Dynamic Nonlinear Statistical Models.
           *Journal of the American Statistical Association*, 92(440),
           1621-1629
        """
        params = self._internal_params(np.asarray(params))
        yhat = np.zeros(self.nobs, dtype=params.dtype)
        xhat = np.zeros((self.nobs, self._k_states_internal), dtype=params.dtype)
        return self._loglike_internal(np.asarray(params), yhat, xhat)

    def _residuals(self, yhat, data=None):
        """Calculates residuals of a prediction"""
        if data is None:
            data = self.endog
        if self.error == 'mul':
            return (data - yhat) / yhat
        else:
            return data - yhat

    def _smooth(self, params):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters

        Returns
        -------
        yhat : pd.Series or np.ndarray
            Predicted values from exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.Series``, else a ``np.ndarray``.
        xhat : pd.DataFrame or np.ndarray
            Internal states of exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.DataFrame``, else a ``np.ndarray``.
        """
        internal_params = self._internal_params(params)
        yhat = np.zeros(self.nobs)
        xhat = np.zeros((self.nobs, self._k_states_internal))
        is_fixed = np.zeros(self._k_params_internal, dtype=np.int64)
        fixed_values = np.empty(self._k_params_internal, dtype=params.dtype)
        self._smoothing_func(internal_params, self.endog, yhat, xhat, is_fixed, fixed_values)
        states = self._get_states(xhat)
        if self.use_pandas:
            _, _, _, index = self._get_prediction_index(0, self.nobs - 1)
            yhat = pd.Series(yhat, index=index)
            statenames = ['level']
            if self.has_trend:
                statenames += ['trend']
            if self.has_seasonal:
                statenames += ['seasonal']
            states = pd.DataFrame(states, index=index, columns=statenames)
        return (yhat, states)

    def smooth(self, params, return_raw=False):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters
        return_raw : bool, optional
            Whether to return only the state space results or the full results
            object. Default is ``False``.

        Returns
        -------
        result : ETSResultsWrapper or tuple
            If ``return_raw=False``, returns a ETSResultsWrapper
            object. Otherwise a tuple of arrays or pandas objects, depending on
            the format of the endog data.
        """
        params = np.asarray(params)
        results = self._smooth(params)
        return self._wrap_results(params, results, return_raw)

    @property
    def _res_classes(self):
        return {'fit': (ETSResults, ETSResultsWrapper)}

    def hessian(self, params, approx_centered=False, approx_complex_step=True, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        approx_centered : bool
            Whether to use a centered scheme for finite difference
            approximation
        approx_complex_step : bool
            Whether to use complex step differentiation for approximation

        Returns
        -------
        hessian : ndarray
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.
        """
        method = kwargs.get('method', 'approx')
        if method == 'approx':
            if approx_complex_step:
                hessian = self._hessian_complex_step(params, **kwargs)
            else:
                hessian = self._hessian_finite_difference(params, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid Hessian calculation method.')
        return hessian

    def score(self, params, approx_centered=False, approx_complex_step=True, **kwargs):
        method = kwargs.get('method', 'approx')
        if method == 'approx':
            if approx_complex_step:
                score = self._score_complex_step(params, **kwargs)
            else:
                score = self._score_finite_difference(params, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid score method.')
        return score

    def update(params, *args, **kwargs):
        ...