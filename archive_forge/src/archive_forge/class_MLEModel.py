from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
class MLEModel(tsbase.TimeSeriesModel):
    """
    State space model for maximum likelihood estimation

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k. Default is no
        exogenous regressors.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    ssm : statsmodels.tsa.statespace.kalman_filter.KalmanFilter
        Underlying state space representation.

    See Also
    --------
    statsmodels.tsa.statespace.mlemodel.MLEResults
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation

    Notes
    -----
    This class wraps the state space model with Kalman filtering to add in
    functionality for maximum likelihood estimation. In particular, it adds
    the concept of updating the state space representation based on a defined
    set of parameters, through the `update` method or `updater` attribute (see
    below for more details on which to use when), and it adds a `fit` method
    which uses a numerical optimizer to select the parameters that maximize
    the likelihood of the model.

    The `start_params` `update` method must be overridden in the
    child class (and the `transform` and `untransform` methods, if needed).
    """

    def __init__(self, endog, k_states, exog=None, dates=None, freq=None, **kwargs):
        super().__init__(endog=endog, exog=exog, dates=dates, freq=freq, missing='none')
        self._init_kwargs = kwargs
        self.endog, self.exog = self.prepare_data()
        self.nobs = self.endog.shape[0]
        self.k_states = k_states
        self.initialize_statespace(**kwargs)
        self._has_fixed_params = False
        self._fixed_params = None
        self._params_index = None
        self._fixed_params_index = None
        self._free_params_index = None

    def prepare_data(self):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(self.data.orig_endog, order='C')
        exog = self.data.orig_exog
        if exog is not None:
            exog = np.array(exog)
        if endog.ndim == 1:
            endog.shape = (endog.shape[0], 1)
        return (endog, exog)

    def initialize_statespace(self, **kwargs):
        """
        Initialize the state space representation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the state space class
            constructor.
        """
        endog = self.endog.T
        self.ssm = SimulationSmoother(endog.shape[0], self.k_states, nobs=endog.shape[1], **kwargs)
        self.ssm.bind(endog)
        self.k_endog = self.ssm.k_endog

    def _get_index_with_final_state(self):
        if self._index_dates:
            if isinstance(self._index, pd.DatetimeIndex):
                index = pd.date_range(start=self._index[0], periods=len(self._index) + 1, freq=self._index.freq)
            elif isinstance(self._index, pd.PeriodIndex):
                index = pd.period_range(start=self._index[0], periods=len(self._index) + 1, freq=self._index.freq)
            else:
                raise NotImplementedError
        elif isinstance(self._index, pd.RangeIndex):
            try:
                start = self._index.start
                stop = self._index.stop
                step = self._index.step
            except AttributeError:
                start = self._index._start
                stop = self._index._stop
                step = self._index._step
            index = pd.RangeIndex(start, stop + step, step)
        elif is_int_index(self._index):
            value = self._index[-1] + 1
            index = pd.Index(self._index.tolist() + [value])
        else:
            raise NotImplementedError
        return index

    def __setitem__(self, key, value):
        return self.ssm.__setitem__(key, value)

    def __getitem__(self, key):
        return self.ssm.__getitem__(key)

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        for key, value in kwds.items():
            if value is None and hasattr(self.ssm, key):
                kwds[key] = getattr(self.ssm, key)
        return kwds

    def clone(self, endog, exog=None, **kwargs):
        """
        Clone state space model with new data and optionally new specification

        Parameters
        ----------
        endog : array_like
            The observed time-series process :math:`y`
        k_states : int
            The dimension of the unobserved state process.
        exog : array_like, optional
            Array of exogenous regressors, shaped nobs x k. Default is no
            exogenous regressors.
        kwargs
            Keyword arguments to pass to the new model class to change the
            model specification.

        Returns
        -------
        model : MLEModel subclass

        Notes
        -----
        This method must be implemented
        """
        raise NotImplementedError('This method is not implemented in the base class and must be set up by each specific model.')

    def _clone_from_init_kwds(self, endog, **kwargs):
        use_kwargs = self._get_init_kwds()
        use_kwargs.update(kwargs)
        if getattr(self, 'k_exog', 0) > 0 and kwargs.get('exog', None) is None:
            raise ValueError('Cloning a model with an exogenous component requires specifying a new exogenous array using the `exog` argument.')
        mod = self.__class__(endog, **use_kwargs)
        return mod

    def set_filter_method(self, filter_method=None, **kwargs):
        """
        Set the filtering method

        The filtering method controls aspects of which Kalman filtering
        approach will be used.

        Parameters
        ----------
        filter_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_filter_method(filter_method, **kwargs)

    def set_inversion_method(self, inversion_method=None, **kwargs):
        """
        Set the inversion method

        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.

        Parameters
        ----------
        inversion_method : int, optional
            Bitmask value to set the inversion method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the inversion method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_inversion_method(inversion_method, **kwargs)

    def set_stability_method(self, stability_method=None, **kwargs):
        """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : int, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_stability_method(stability_method, **kwargs)

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : int, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_conserve_memory(conserve_memory, **kwargs)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        """
        Set the smoother output

        The smoother can produce several types of results. The smoother output
        variable controls which are calculated and returned.

        Parameters
        ----------
        smoother_output : int, optional
            Bitmask value to set the smoother output to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the smoother output by
            setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanSmoother` class for details.
        """
        self.ssm.set_smoother_output(smoother_output, **kwargs)

    def initialize_known(self, initial_state, initial_state_cov):
        """Initialize known"""
        self.ssm.initialize_known(initial_state, initial_state_cov)

    def initialize_approximate_diffuse(self, variance=None):
        """Initialize approximate diffuse"""
        self.ssm.initialize_approximate_diffuse(variance)

    def initialize_stationary(self):
        """Initialize stationary"""
        self.ssm.initialize_stationary()

    @property
    def initialization(self):
        return self.ssm.initialization

    @initialization.setter
    def initialization(self, value):
        self.ssm.initialization = value

    @property
    def initial_variance(self):
        return self.ssm.initial_variance

    @initial_variance.setter
    def initial_variance(self, value):
        self.ssm.initial_variance = value

    @property
    def loglikelihood_burn(self):
        return self.ssm.loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        self.ssm.loglikelihood_burn = value

    @property
    def tolerance(self):
        return self.ssm.tolerance

    @tolerance.setter
    def tolerance(self, value):
        self.ssm.tolerance = value

    def _validate_can_fix_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError('Invalid parameter name passed: "%s".' % param_name)

    @contextlib.contextmanager
    def fix_params(self, params):
        """
        Fix parameters to specific values (context manager)

        Parameters
        ----------
        params : dict
            Dictionary describing the fixed parameter values, of the form
            `param_name: fixed_value`. See the `param_names` property for valid
            parameter names.

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> with mod.fix_params({'ar.L1': 0.5}):
                res = mod.fit()
        """
        k_params = len(self.param_names)
        if self._fixed_params is None:
            self._fixed_params = {}
            self._params_index = dict(zip(self.param_names, np.arange(k_params)))
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index
        all_fixed_param_names = set(params.keys()) | set(self._fixed_params.keys())
        self._validate_can_fix_params(all_fixed_param_names)
        self._fixed_params.update(params)
        self._fixed_params = {name: self._fixed_params[name] for name in self.param_names if name in self._fixed_params}
        self._has_fixed_params = True
        self._fixed_params_index = [self._params_index[key] for key in self._fixed_params.keys()]
        self._free_params_index = list(set(np.arange(k_params)).difference(self._fixed_params_index))
        try:
            yield
        finally:
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

    def fit(self, start_params=None, transformed=True, includes_fixed=False, cov_type=None, cov_kwds=None, method='lbfgs', maxiter=50, full_output=1, disp=5, callback=None, return_params=False, optim_score=None, optim_complex_step=None, optim_hessian=None, flags=None, low_memory=False, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'opg' unless memory conservation is used to avoid
            computing the loglikelihood values for each observation, in which
            case the default is 'approx'.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson
            - 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
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
        optim_score : {'harvey', 'approx'} or None, optional
            The method by which the score vector is calculated. 'harvey' uses
            the method from Harvey (1989), 'approx' uses either finite
            difference or complex step differentiation depending upon the
            value of `optim_complex_step`, and None uses the built-in gradient
            approximation of the optimizer. Default is None. This keyword is
            only relevant if the optimization method uses the score.
        optim_complex_step : bool, optional
            Whether or not to use complex step differentiation when
            approximating the score; if False, finite difference approximation
            is used. Default is True. This keyword is only relevant if
            `optim_score` is set to 'harvey' or 'approx'.
        optim_hessian : {'opg','oim','approx'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'approx' uses numerical
            approximation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including smoothed results and in-sample
            prediction), although out-of-sample forecasting is possible.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results
            Results object holding results from fitting a state space model.

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        statsmodels.tsa.statespace.mlemodel.MLEResults
        statsmodels.tsa.statespace.structural.UnobservedComponentsResults
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
            includes_fixed = True
        if optim_score is None and method == 'lbfgs':
            kwargs.setdefault('approx_grad', True)
            kwargs.setdefault('epsilon', 1e-05)
        elif optim_score is None:
            optim_score = 'approx'
        if optim_complex_step is None:
            optim_complex_step = not self.ssm._complex_endog
        elif optim_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data or parameters are complex.')
        start_params = self.handle_params(start_params, transformed=True, includes_fixed=includes_fixed)
        if transformed:
            start_params = self.untransform_params(start_params)
        if self._has_fixed_params:
            start_params = start_params[self._free_params_index]
        if self._has_fixed_params and len(start_params) == 0:
            mlefit = Bunch(params=[], mle_retvals=None, mle_settings=None)
        else:
            disallow = ('concentrate_scale', 'enforce_stationarity', 'enforce_invertibility')
            kwargs = {k: v for k, v in kwargs.items() if k not in disallow}
            if flags is None:
                flags = {}
            flags.update({'transformed': False, 'includes_fixed': False, 'score_method': optim_score, 'approx_complex_step': optim_complex_step})
            if optim_hessian is not None:
                flags['hessian_method'] = optim_hessian
            fargs = (flags,)
            mlefit = super().fit(start_params, method=method, fargs=fargs, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, skip_hessian=True, **kwargs)
        if return_params:
            return self.handle_params(mlefit.params, transformed=False, includes_fixed=False)
        else:
            if low_memory:
                conserve_memory = self.ssm.conserve_memory
                self.ssm.set_conserve_memory(MEMORY_CONSERVE)
            if self.ssm.memory_no_predicted or self.ssm.memory_no_gain or self.ssm.memory_no_smoothing:
                func = self.filter
            else:
                func = self.smooth
            res = func(mlefit.params, transformed=False, includes_fixed=False, cov_type=cov_type, cov_kwds=cov_kwds)
            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings
            if low_memory:
                self.ssm.set_conserve_memory(conserve_memory)
            return res

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """
        Fit the model with some parameters subject to equality constraints.

        Parameters
        ----------
        constraints : dict
            Dictionary of constraints, of the form `param_name: fixed_value`.
            See the `param_names` property for valid parameter names.
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the remaining parameters.

        Returns
        -------
        results : Results instance

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> res = mod.fit_constrained({'ar.L1': 0.5})
        """
        with self.fix_params(constraints):
            res = self.fit(start_params, **fit_kwds)
        return res

    @property
    def _res_classes(self):
        return {'fit': (MLEResults, MLEResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None, cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds
            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]
            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def filter(self, params, transformed=True, includes_fixed=False, complex_step=False, cov_type=None, cov_kwds=None, return_ssm=False, results_class=None, results_wrapper_class=None, low_memory=False, **kwargs):
        """
        Kalman filtering

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including in-sample prediction), although
            out-of-sample forecasting is possible. Default is False.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True, complex_step=complex_step)
        self.data.param_names = self.param_names
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        if low_memory:
            kwargs['conserve_memory'] = MEMORY_CONSERVE
        result = self.ssm.filter(complex_step=complex_step, **kwargs)
        return self._wrap_results(params, result, return_ssm, cov_type, cov_kwds, results_class, results_wrapper_class)

    def smooth(self, params, transformed=True, includes_fixed=False, complex_step=False, cov_type=None, cov_kwds=None, return_ssm=False, results_class=None, results_wrapper_class=None, **kwargs):
        """
        Kalman smoothing

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True, complex_step=complex_step)
        self.data.param_names = self.param_names
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        result = self.ssm.smooth(complex_step=complex_step, **kwargs)
        return self._wrap_results(params, result, return_ssm, cov_type, cov_kwds, results_class, results_wrapper_class)
    _loglike_param_names = ['transformed', 'includes_fixed', 'complex_step']
    _loglike_param_defaults = [True, False, False]

    def loglike(self, params, *args, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        See Also
        --------
        update : modifies the internal state of the state space model to
                 reflect new params

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        transformed, includes_fixed, complex_step, kwargs = _handle_args(MLEModel._loglike_param_names, MLEModel._loglike_param_defaults, *args, **kwargs)
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True, complex_step=complex_step)
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
        return loglike

    def loglikeobs(self, params, transformed=True, includes_fixed=False, complex_step=False, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        See Also
        --------
        update : modifies the internal state of the Model to reflect new params

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        self.update(params, transformed=True, includes_fixed=True, complex_step=complex_step)
        return self.ssm.loglikeobs(complex_step=complex_step, **kwargs)

    def simulation_smoother(self, simulation_output=None, **kwargs):
        """
        Retrieve a simulation smoother for the state space model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        return self.ssm.simulation_smoother(simulation_output=simulation_output, **kwargs)

    def _forecasts_error_partial_derivatives(self, params, transformed=True, includes_fixed=False, approx_complex_step=None, approx_centered=False, res=None, **kwargs):
        params = np.array(params, ndmin=1)
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError('Cannot use complex-step approximations to calculate the observed_information_matrix with untransformed parameters.')
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        if res is None:
            self.update(params, transformed=transformed, includes_fixed=includes_fixed, complex_step=approx_complex_step)
            res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
        n = len(params)
        partials_forecasts_error = np.zeros((self.k_endog, self.nobs, n))
        partials_forecasts_error_cov = np.zeros((self.k_endog, self.k_endog, self.nobs, n))
        if approx_complex_step:
            epsilon = _get_epsilon(params, 2, None, n)
            increments = np.identity(n) * 1j * epsilon
            for i, ih in enumerate(increments):
                self.update(params + ih, transformed=transformed, includes_fixed=includes_fixed, complex_step=True)
                _res = self.ssm.filter(complex_step=True, **kwargs)
                partials_forecasts_error[:, :, i] = _res.forecasts_error.imag / epsilon[i]
                partials_forecasts_error_cov[:, :, :, i] = _res.forecasts_error_cov.imag / epsilon[i]
        elif not approx_centered:
            epsilon = _get_epsilon(params, 2, None, n)
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]
                self.update(params + ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
                _res = self.ssm.filter(complex_step=False, **kwargs)
                partials_forecasts_error[:, :, i] = (_res.forecasts_error - res.forecasts_error) / epsilon[i]
                partials_forecasts_error_cov[:, :, :, i] = (_res.forecasts_error_cov - res.forecasts_error_cov) / epsilon[i]
                ei[i] = 0.0
        else:
            epsilon = _get_epsilon(params, 3, None, n) / 2.0
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]
                self.update(params + ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
                _res1 = self.ssm.filter(complex_step=False, **kwargs)
                self.update(params - ei, transformed=transformed, includes_fixed=includes_fixed, complex_step=False)
                _res2 = self.ssm.filter(complex_step=False, **kwargs)
                partials_forecasts_error[:, :, i] = (_res1.forecasts_error - _res2.forecasts_error) / (2 * epsilon[i])
                partials_forecasts_error_cov[:, :, :, i] = (_res1.forecasts_error_cov - _res2.forecasts_error_cov) / (2 * epsilon[i])
                ei[i] = 0.0
        return (partials_forecasts_error, partials_forecasts_error_cov)

    def observed_information_matrix(self, params, transformed=True, includes_fixed=False, approx_complex_step=None, approx_centered=False, **kwargs):
        """
        Observed information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), which shows that the information
        matrix only depends on terms from the gradient. This implementation is
        partially analytic and partially numeric approximation, therefore,
        because it uses the analytic formula for the information matrix, with
        numerically computed elements of the gradient.

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.
        """
        params = np.array(params, ndmin=1)
        n = len(params)
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError('Cannot use complex-step approximations to calculate the observed_information_matrix with untransformed parameters.')
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True, complex_step=approx_complex_step)
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
        dtype = self.ssm.dtype
        inv_forecasts_error_cov = res.forecasts_error_cov.copy()
        partials_forecasts_error, partials_forecasts_error_cov = self._forecasts_error_partial_derivatives(params, transformed=transformed, includes_fixed=includes_fixed, approx_complex_step=approx_complex_step, approx_centered=approx_centered, res=res, **kwargs)
        tmp = np.zeros((self.k_endog, self.k_endog, self.nobs, n), dtype=dtype)
        information_matrix = np.zeros((n, n), dtype=dtype)
        d = np.maximum(self.ssm.loglikelihood_burn, res.nobs_diffuse)
        for t in range(d, self.nobs):
            inv_forecasts_error_cov[:, :, t] = np.linalg.inv(res.forecasts_error_cov[:, :, t])
            for i in range(n):
                tmp[:, :, t, i] = np.dot(inv_forecasts_error_cov[:, :, t], partials_forecasts_error_cov[:, :, t, i])
            for i in range(n):
                for j in range(n):
                    information_matrix[i, j] += 0.5 * np.trace(np.dot(tmp[:, :, t, i], tmp[:, :, t, j]))
                    information_matrix[i, j] += np.inner(partials_forecasts_error[:, t, i], np.dot(inv_forecasts_error_cov[:, :, t], partials_forecasts_error[:, t, j]))
        return information_matrix / (self.nobs - self.ssm.loglikelihood_burn)

    def opg_information_matrix(self, params, transformed=True, includes_fixed=False, approx_complex_step=None, **kwargs):
        """
        Outer product of gradients information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional arguments to the `loglikeobs` method.

        References
        ----------
        Berndt, Ernst R., Bronwyn Hall, Robert Hall, and Jerry Hausman. 1974.
        Estimation and Inference in Nonlinear Structural Models.
        NBER Chapters. National Bureau of Economic Research, Inc.
        """
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError('Cannot use complex-step approximations to calculate the observed_information_matrix with untransformed parameters.')
        score_obs = self.score_obs(params, transformed=transformed, includes_fixed=includes_fixed, approx_complex_step=approx_complex_step, **kwargs).transpose()
        return np.inner(score_obs, score_obs) / (self.nobs - self.ssm.loglikelihood_burn)

    def _score_complex_step(self, params, **kwargs):
        epsilon = _get_epsilon(params, 2.0, None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        return approx_fprime_cs(params, self.loglike, epsilon=epsilon, kwargs=kwargs)

    def _score_finite_difference(self, params, approx_centered=False, **kwargs):
        kwargs['transformed'] = True
        return approx_fprime(params, self.loglike, kwargs=kwargs, centered=approx_centered)

    def _score_harvey(self, params, approx_complex_step=True, **kwargs):
        score_obs = self._score_obs_harvey(params, approx_complex_step=approx_complex_step, **kwargs)
        return np.sum(score_obs, axis=0)

    def _score_obs_harvey(self, params, approx_complex_step=True, approx_centered=False, includes_fixed=False, **kwargs):
        """
        Score

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), section 3.4.5

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.
        """
        params = np.array(params, ndmin=1)
        n = len(params)
        self.update(params, transformed=True, includes_fixed=includes_fixed, complex_step=approx_complex_step)
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        if 'transformed' in kwargs:
            del kwargs['transformed']
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
        partials_forecasts_error, partials_forecasts_error_cov = self._forecasts_error_partial_derivatives(params, transformed=True, includes_fixed=includes_fixed, approx_complex_step=approx_complex_step, approx_centered=approx_centered, res=res, **kwargs)
        partials = np.zeros((self.nobs, n))
        k_endog = self.k_endog
        for t in range(self.nobs):
            inv_forecasts_error_cov = np.linalg.inv(res.forecasts_error_cov[:, :, t])
            for i in range(n):
                partials[t, i] += np.trace(np.dot(np.dot(inv_forecasts_error_cov, partials_forecasts_error_cov[:, :, t, i]), np.eye(k_endog) - np.dot(inv_forecasts_error_cov, np.outer(res.forecasts_error[:, t], res.forecasts_error[:, t]))))
                partials[t, i] += 2 * np.dot(partials_forecasts_error[:, t, i], np.dot(inv_forecasts_error_cov, res.forecasts_error[:, t]))
        return -partials / 2.0
    _score_param_names = ['transformed', 'includes_fixed', 'score_method', 'approx_complex_step', 'approx_centered']
    _score_param_defaults = [True, False, 'approx', None, False]

    def score(self, params, *args, **kwargs):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        *args
            Additional positional arguments to the `loglike` method.
        **kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        score : ndarray
            Score, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglike` method.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        transformed, includes_fixed, method, approx_complex_step, approx_centered, kwargs = _handle_args(MLEModel._score_param_names, MLEModel._score_param_defaults, *args, **kwargs)
        if 'method' in kwargs:
            method = kwargs.pop('method')
        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data or parameters are complex.')
        out = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed, return_jacobian=not transformed)
        if transformed:
            params = out
        else:
            params, transform_score = out
        if method == 'harvey':
            kwargs['includes_fixed'] = True
            score = self._score_harvey(params, approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            kwargs['includes_fixed'] = True
            score = self._score_complex_step(params, **kwargs)
        elif method == 'approx':
            kwargs['includes_fixed'] = True
            score = self._score_finite_difference(params, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid score method.')
        if not transformed:
            score = np.dot(transform_score, score)
        if self._has_fixed_params and (not includes_fixed):
            score = score[self._free_params_index]
        return score

    def score_obs(self, params, method='approx', transformed=True, includes_fixed=False, approx_complex_step=None, approx_centered=False, **kwargs):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        **kwargs
            Additional arguments to the `loglike` method.

        Returns
        -------
        score : ndarray
            Score per observation, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglikeobs` method.
        """
        if not transformed and approx_complex_step:
            raise ValueError('Cannot use complex-step approximations to calculate the score at each observation with untransformed parameters.')
        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data or parameters are complex.')
        params = self.handle_params(params, transformed=True, includes_fixed=includes_fixed)
        kwargs['transformed'] = transformed
        kwargs['includes_fixed'] = True
        if method == 'harvey':
            score = self._score_obs_harvey(params, approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            epsilon = _get_epsilon(params, 2.0, None, len(params))
            kwargs['complex_step'] = True
            score = approx_fprime_cs(params, self.loglikeobs, epsilon=epsilon, kwargs=kwargs)
        elif method == 'approx':
            score = approx_fprime(params, self.loglikeobs, kwargs=kwargs, centered=approx_centered)
        else:
            raise NotImplementedError('Invalid scoreobs method.')
        return score
    _hessian_param_names = ['transformed', 'hessian_method', 'approx_complex_step', 'approx_centered']
    _hessian_param_defaults = [True, 'approx', None, False]

    def hessian(self, params, *args, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        *args
            Additional positional arguments to the `loglike` method.
        **kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        hessian : ndarray
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        transformed, method, approx_complex_step, approx_centered, kwargs = _handle_args(MLEModel._hessian_param_names, MLEModel._hessian_param_defaults, *args, **kwargs)
        if 'method' in kwargs:
            method = kwargs.pop('method')
        if not transformed and approx_complex_step:
            raise ValueError('Cannot use complex-step approximations to calculate the hessian with untransformed parameters.')
        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data or parameters are complex.')
        if method == 'oim':
            hessian = self._hessian_oim(params, transformed=transformed, approx_complex_step=approx_complex_step, approx_centered=approx_centered, **kwargs)
        elif method == 'opg':
            hessian = self._hessian_opg(params, transformed=transformed, approx_complex_step=approx_complex_step, approx_centered=approx_centered, **kwargs)
        elif method == 'approx' and approx_complex_step:
            hessian = self._hessian_complex_step(params, transformed=transformed, **kwargs)
        elif method == 'approx':
            hessian = self._hessian_finite_difference(params, transformed=transformed, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid Hessian calculation method.')
        return hessian

    def _hessian_oim(self, params, **kwargs):
        """
        Hessian matrix computed using the Harvey (1989) information matrix
        """
        return -self.observed_information_matrix(params, **kwargs)

    def _hessian_opg(self, params, **kwargs):
        """
        Hessian matrix computed using the outer product of gradients
        information matrix
        """
        return -self.opg_information_matrix(params, **kwargs)

    def _hessian_finite_difference(self, params, approx_centered=False, **kwargs):
        params = np.array(params, ndmin=1)
        warnings.warn('Calculation of the Hessian using finite differences is usually subject to substantial approximation errors.', PrecisionWarning)
        if not approx_centered:
            epsilon = _get_epsilon(params, 3, None, len(params))
        else:
            epsilon = _get_epsilon(params, 4, None, len(params)) / 2
        hessian = approx_fprime(params, self._score_finite_difference, epsilon=epsilon, kwargs=kwargs, centered=approx_centered)
        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    def _hessian_complex_step(self, params, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
        epsilon = _get_epsilon(params, 3.0, None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        hessian = approx_hess_cs(params, self.loglike, epsilon=epsilon, kwargs=kwargs)
        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        if hasattr(self, '_start_params'):
            return self._start_params
        else:
            raise NotImplementedError

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        if hasattr(self, '_param_names'):
            return self._param_names
        else:
            try:
                names = ['param.%d' % i for i in range(len(self.start_params))]
            except NotImplementedError:
                names = []
            return names

    @property
    def state_names(self):
        """
        (list of str) List of human readable names for unobserved states.
        """
        if hasattr(self, '_state_names'):
            return self._state_names
        else:
            names = ['state.%d' % i for i in range(self.k_states)]
        return names

    def transform_jacobian(self, unconstrained, approx_centered=False):
        """
        Jacobian matrix for the parameter transformation function

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Returns
        -------
        jacobian : ndarray
            Jacobian matrix of the transformation, evaluated at `unconstrained`

        See Also
        --------
        transform_params

        Notes
        -----
        This is a numerical approximation using finite differences. Note that
        in general complex step methods cannot be used because it is not
        guaranteed that the `transform_params` method is a real function (e.g.
        if Cholesky decomposition is used).
        """
        return approx_fprime(unconstrained, self.transform_params, centered=approx_centered)

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(unconstrained, ndmin=1)

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(constrained, ndmin=1)

    def handle_params(self, params, transformed=True, includes_fixed=False, return_jacobian=False):
        """
        Ensure model parameters satisfy shape and other requirements
        """
        params = np.array(params, ndmin=1)
        if np.issubdtype(params.dtype, np.integer):
            params = params.astype(np.float64)
        if not includes_fixed and self._has_fixed_params:
            k_params = len(self.param_names)
            new_params = np.zeros(k_params, dtype=params.dtype) * np.nan
            new_params[self._free_params_index] = params
            params = new_params
        if not transformed:
            if not includes_fixed and self._has_fixed_params:
                params[self._fixed_params_index] = list(self._fixed_params.values())
            if return_jacobian:
                transform_score = self.transform_jacobian(params)
            params = self.transform_params(params)
        if not includes_fixed and self._has_fixed_params:
            params[self._fixed_params_index] = list(self._fixed_params.values())
        return (params, transform_score) if return_jacobian else params

    def update(self, params, transformed=True, includes_fixed=False, complex_step=False):
        """
        Update the parameters of the model

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Since Model is a base class, this method should be overridden by
        subclasses to perform actual updating steps.
        """
        return self.handle_params(params=params, transformed=transformed, includes_fixed=includes_fixed)

    def _validate_out_of_sample_exog(self, exog, out_of_sample):
        """
        Validate given `exog` as satisfactory for out-of-sample operations

        Parameters
        ----------
        exog : array_like or None
            New observations of exogenous regressors, if applicable.
        out_of_sample : int
            Number of new observations required.

        Returns
        -------
        exog : array or None
            A numpy array of shape (out_of_sample, k_exog) if the model
            contains an `exog` component, or None if it does not.
        """
        k_exog = getattr(self, 'k_exog', 0)
        if out_of_sample and k_exog > 0:
            if exog is None:
                raise ValueError('Out-of-sample operations in a model with a regression component require additional exogenous values via the `exog` argument.')
            exog = np.array(exog)
            required_exog_shape = (out_of_sample, self.k_exog)
            try:
                exog = exog.reshape(required_exog_shape)
            except ValueError:
                raise ValueError('Provided exogenous values are not of the appropriate shape. Required %s, got %s.' % (str(required_exog_shape), str(exog.shape)))
        elif k_exog > 0 and exog is not None:
            exog = None
            warnings.warn('Exogenous array provided, but additional data is not required. `exog` argument ignored.', ValueWarning)
        return exog

    def _get_extension_time_varying_matrices(self, params, exog, out_of_sample, extend_kwargs=None, transformed=True, includes_fixed=False, **kwargs):
        """
        Get updated time-varying state space system matrices

        Parameters
        ----------
        params : array_like
            Array of parameters used to construct the time-varying system
            matrices.
        exog : array_like or None
            New observations of exogenous regressors, if applicable.
        out_of_sample : int
            Number of new observations required.
        extend_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the state space model
            constructor. For example, for an SARIMAX state space model, this
            could be used to pass the `concentrate_scale=True` keyword
            argument. Any arguments that are not explicitly set in this
            dictionary will be copied from the current model instance.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        """
        exog = self._validate_out_of_sample_exog(exog, out_of_sample)
        if extend_kwargs is None:
            extend_kwargs = {}
        if getattr(self, 'k_trend', 0) > 0 and hasattr(self, 'trend_offset'):
            extend_kwargs.setdefault('trend_offset', self.trend_offset + self.nobs)
        mod_extend = self.clone(endog=np.zeros((out_of_sample, self.k_endog)), exog=exog, **extend_kwargs)
        mod_extend.update(params, transformed=transformed, includes_fixed=includes_fixed)
        for name in self.ssm.shapes.keys():
            if name == 'obs' or name in kwargs:
                continue
            original = getattr(self.ssm, name)
            extended = getattr(mod_extend.ssm, name)
            so = original.shape[-1]
            se = extended.shape[-1]
            if (so > 1 or se > 1) or (so == 1 and self.nobs == 1 and np.any(original[..., 0] != extended[..., 0])):
                kwargs[name] = extended[..., -out_of_sample:]
        return kwargs

    def simulate(self, params, nsimulations, measurement_shocks=None, state_shocks=None, initial_state=None, anchor=None, repetitions=None, exog=None, extend_model=None, extend_kwargs=None, transformed=True, includes_fixed=False, pretransformed_measurement_shocks=True, pretransformed_state_shocks=True, pretransformed_initial_state=True, random_state=None, **kwargs):
        """
        Simulate a new time series following the state space model

        Parameters
        ----------
        params : array_like
            Array of parameters to use in constructing the state space
            representation to use when simulating.
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number of observations.
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the initial state vector to use in
            simulation, which should be shaped (`k_states` x 1), where
            `k_states` is the same as in the state space model. If unspecified,
            but the model has been initialized, then that initialization is
            used. This must be specified if `anchor` is anything other than
            "start" or 0 (or else you can use the `simulate` method on a
            results object rather than on the model object).
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        pretransformed_measurement_shocks : bool, optional
            If `measurement_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `obs_cov` covariance matrix. Default is True.
        pretransformed_state_shocks : bool, optional
            If `state_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `state_cov` covariance matrix. Default is
            True.
        pretransformed_initial_state : bool, optional
            If `initial_state` is provided, this flag indicates whether it
            should be directly used as the initial_state. If False, then it is
            assumed to contain draws from the standard Normal distribution that
            must be transformed using the `initial_state_cov` covariance
            matrix. Default is True.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the
            class:``~numpy.random.RandomState`` singleton is used.
            If `seed` is an int, a new class:``~numpy.random.RandomState``
            instance is used, seeded with `seed`.
            If `seed` is already a class:``~numpy.random.Generator`` or
            class:``~numpy.random.RandomState`` instance then that instance is
            used.

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If `repetitions=None`, then it
            will be shaped (nsimulations x k_endog) or (nsimulations,) if
            `k_endog=1`. Otherwise it will be shaped
            (nsimulations x k_endog x repetitions). If the model was given
            Pandas input then the output will be a Pandas object. If
            `k_endog > 1` and `repetitions` is not None, then the output will
            be a Pandas DataFrame that has a MultiIndex for the columns, with
            the first level containing the names of the `endog` variables and
            the second level containing the repetition number.

        See Also
        --------
        impulse_responses
            Impulse response functions
        """
        self.update(params, transformed=transformed, includes_fixed=includes_fixed)
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start
        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation outside of the sample.')
        if iloc > 0 and initial_state is None:
            raise ValueError('If `anchor` is after the start of the sample, must provide a value for `initial_state`.')
        out_of_sample = max(iloc + nsimulations - self.nobs, 0)
        if extend_model is None:
            extend_model = self.exog is not None or not self.ssm.time_invariant
        if out_of_sample and extend_model:
            kwargs = self._get_extension_time_varying_matrices(params, exog, out_of_sample, extend_kwargs, transformed=transformed, includes_fixed=includes_fixed, **kwargs)
        if initial_state is not None:
            initial_state = np.array(initial_state)
            if initial_state.ndim < 2:
                initial_state = np.atleast_2d(initial_state).T
        end = min(self.nobs, iloc + nsimulations)
        nextend = iloc + nsimulations - end
        sim_model = self.ssm.extend(np.zeros((nextend, self.k_endog)), start=iloc, end=end, **kwargs)
        _repetitions = 1 if repetitions is None else repetitions
        sim = np.zeros((nsimulations, self.k_endog, _repetitions))
        simulator = None
        for i in range(_repetitions):
            initial_state_variates = None
            if initial_state is not None:
                if initial_state.shape[1] == 1:
                    initial_state_variates = initial_state[:, 0]
                else:
                    initial_state_variates = initial_state[:, i]
            out, _, simulator = sim_model.simulate(nsimulations, measurement_shocks, state_shocks, initial_state_variates, pretransformed_measurement_shocks=pretransformed_measurement_shocks, pretransformed_state_shocks=pretransformed_state_shocks, pretransformed_initial_state=pretransformed_initial_state, simulator=simulator, return_simulator=True, random_state=random_state)
            sim[:, :, i] = out
        use_pandas = isinstance(self.data, PandasData)
        index = None
        if use_pandas:
            _, _, _, index = self._get_prediction_index(iloc, iloc + nsimulations - 1)
        if repetitions is None:
            if self.k_endog == 1:
                sim = sim[:, 0, 0]
                if use_pandas:
                    sim = pd.Series(sim, index=index, name=self.endog_names)
            else:
                sim = sim[:, :, 0]
                if use_pandas:
                    sim = pd.DataFrame(sim, index=index, columns=self.endog_names)
        elif use_pandas:
            shape = sim.shape
            endog_names = self.endog_names
            if not isinstance(endog_names, list):
                endog_names = [endog_names]
            columns = pd.MultiIndex.from_product([endog_names, np.arange(shape[2])])
            sim = pd.DataFrame(sim.reshape(shape[0], shape[1] * shape[2]), index=index, columns=columns)
        return sim

    def impulse_responses(self, params, steps=1, impulse=0, orthogonalized=False, cumulative=False, anchor=None, exog=None, extend_model=None, extend_kwargs=None, transformed=True, includes_fixed=False, **kwargs):
        """
        Impulse response function

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that for time-invariant models, the initial
            impulse is not counted as a step, so if `steps=1`, the output will
            have 2 entries.
        impulse : int, str or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. If a str, it indicates which column of df
            the unit (1) impulse is given.
            Alternatively, a custom impulse vector may be provided; must be
            shaped `k_posdef x 1`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        anchor : int, str, or datetime, optional
            Time point within the sample for the state innovation impulse. Type
            depends on the index of the given `endog` in the model. Two special
            cases are the strings 'start' and 'end', which refer to setting the
            impulse at the first and last points of the sample, respectively.
            Integer values can run from 0 to `nobs - 1`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        exog : array_like, optional
            New observations of exogenous regressors for our-of-sample periods,
            if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        **kwargs
            If the model has time-varying design or transition matrices and the
            combination of `anchor` and `steps` implies creating impulse
            responses for the out-of-sample period, then these matrices must
            have updated values provided for the out-of-sample steps. For
            example, if `design` is a time-varying component, `nobs` is 10,
            `anchor=1`, and `steps` is 15, a (`k_endog` x `k_states` x 7)
            matrix must be provided with the new design matrix values.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. For a time-invariant model, the
            impulse responses are given for `steps + 1` elements (this gives
            the "initial impulse" followed by `steps` responses for the
            important cases of VAR and SARIMAX models), while for time-varying
            models the impulse responses are only given for `steps` elements
            (to avoid having to unexpectedly provide updated time-varying
            matrices).

        See Also
        --------
        simulate
            Simulate a time series according to the given state space model,
            optionally with specified series for the innovations.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        TODO: add an option to allow changing the ordering for the
              orthogonalized option. Will require permuting matrices when
              constructing the extended model.
        """
        self.update(params, transformed=transformed, includes_fixed=includes_fixed)
        additional_steps = 0
        if self.ssm._design.shape[2] == 1 and self.ssm._transition.shape[2] == 1 and (self.ssm._selection.shape[2] == 1):
            additional_steps = 1
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs - 1
        else:
            iloc, _, _ = self._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start
        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc >= self.nobs:
            raise ValueError('Cannot anchor impulse responses outside of the sample.')
        time_invariant = self.ssm._design.shape[2] == self.ssm._obs_cov.shape[2] == self.ssm._transition.shape[2] == self.ssm._selection.shape[2] == self.ssm._state_cov.shape[2] == 1
        out_of_sample = max(iloc + (steps + additional_steps + 1) - self.nobs, 0)
        if extend_model is None:
            extend_model = self.exog is not None and (not time_invariant)
        if out_of_sample and extend_model:
            kwargs = self._get_extension_time_varying_matrices(params, exog, out_of_sample, extend_kwargs, transformed=transformed, includes_fixed=includes_fixed, **kwargs)
        end = min(self.nobs, iloc + steps + additional_steps)
        nextend = iloc + (steps + additional_steps + 1) - end
        if 'obs_intercept' not in kwargs and self.ssm._obs_intercept.shape[1] > 1:
            kwargs['obs_intercept'] = np.zeros((self.k_endog, nextend))
        if 'state_intercept' not in kwargs and self.ssm._state_intercept.shape[1] > 1:
            kwargs['state_intercept'] = np.zeros((self.k_states, nextend))
        if 'obs_cov' not in kwargs and self.ssm._obs_cov.shape[2] > 1:
            kwargs['obs_cov'] = np.zeros((self.k_endog, self.k_endog, nextend))
        if 'state_cov' not in kwargs and self.ssm._state_cov.shape[2] > 1:
            tmp = np.zeros((self.ssm.k_posdef, self.ssm.k_posdef, nextend))
            tmp[:] = self['state_cov', :, :, iloc:iloc + 1]
            kwargs['state_cov'] = tmp
        if 'selection' not in kwargs and self.ssm._selection.shape[2] > 1:
            tmp = np.zeros((self.k_states, self.ssm.k_posdef, nextend))
            tmp[:] = self['selection', :, :, iloc:iloc + 1]
            kwargs['selection'] = tmp
        sim_model = self.ssm.extend(np.empty((nextend, self.k_endog)), start=iloc, end=end, **kwargs)
        use_pandas = isinstance(self.data, PandasData)
        if type(impulse) is str:
            if not use_pandas:
                raise ValueError('Endog must be pd.DataFrame.')
            impulse = self.endog_names.index(impulse)
        irfs = sim_model.impulse_responses(steps, impulse, orthogonalized, cumulative)
        if irfs.shape[1] == 1:
            irfs = irfs[:, 0]
        if use_pandas:
            if self.k_endog == 1:
                irfs = pd.Series(irfs, name=self.endog_names)
            else:
                irfs = pd.DataFrame(irfs, columns=self.endog_names)
        return irfs

    @classmethod
    def from_formula(cls, formula, data, subset=None):
        """
        Not implemented for state space models
        """
        raise NotImplementedError