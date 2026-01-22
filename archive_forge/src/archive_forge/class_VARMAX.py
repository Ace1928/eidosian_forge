import contextlib
from warnings import warn
import pandas as pd
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
class VARMAX(MLEModel):
    """
    Vector Autoregressive Moving Average with eXogenous regressors model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`, , shaped nobs x k_endog.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is a constant trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):

    .. math::

        y_t = A(t) + A_1 y_{t-1} + \\dots + A_p y_{t-p} + B x_t + \\epsilon_t +
        M_1 \\epsilon_{t-1} + \\dots M_q \\epsilon_{t-q}

    where :math:`\\epsilon_t \\sim N(0, \\Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
    where the variables are measured with error.

    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\\{A_i, M_j\\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more information. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, exog=None, order=(1, 0), trend='c', error_cov_type='unstructured', measurement_error=False, enforce_stationarity=True, enforce_invertibility=True, trend_offset=1, **kwargs):
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.order = order
        self.k_ar = int(order[0])
        self.k_ma = int(order[1])
        if error_cov_type not in ['diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type specification.')
        if self.k_ar == 0 and self.k_ma == 0:
            raise ValueError('Invalid VARMAX(p,q) specification; at least one p,q must be greater than zero.')
        if self.k_ar > 0 and self.k_ma > 0:
            warn('Estimation of VARMA(p,q) models is not generically robust, due especially to identification issues.', EstimationWarning)
        self.trend = trend
        self.trend_offset = trend_offset
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._trend_is_const = self.polynomial_trend.size == 1 and self.polynomial_trend[0] == 1
        self.k_exog, exog = prepare_exog(exog)
        self.mle_regression = self.k_exog > 0
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)
        _min_k_ar = max(self.k_ar, 1)
        self._k_order = _min_k_ar + self.k_ma
        k_endog = endog.shape[1]
        k_posdef = k_endog
        k_states = k_endog * self._k_order
        kwargs.setdefault('initialization', 'stationary')
        kwargs.setdefault('inversion_method', INVERT_UNIVARIATE | SOLVE_LU)
        super().__init__(endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        if self.k_exog > 0 or (self.k_trend > 0 and (not self._trend_is_const)):
            self.ssm._time_invariant = False
        self.parameters = {}
        self.parameters['trend'] = self.k_endog * self.k_trend
        self.parameters['ar'] = self.k_endog ** 2 * self.k_ar
        self.parameters['ma'] = self.k_endog ** 2 * self.k_ma
        self.parameters['regression'] = self.k_endog * self.k_exog
        if self.error_cov_type == 'diagonal':
            self.parameters['state_cov'] = self.k_endog
        elif self.error_cov_type == 'unstructured':
            self.parameters['state_cov'] = int(self.k_endog * (self.k_endog + 1) / 2)
        self.parameters['obs_cov'] = self.k_endog * self.measurement_error
        self.k_params = sum(self.parameters.values())
        trend_data = prepare_trend_data(self.polynomial_trend, self.k_trend, self.nobs + 1, offset=self.trend_offset)
        self._trend_data = trend_data[:-1]
        self._final_trend = trend_data[-1:]
        if self.k_trend > 0 and (not self._trend_is_const) or self.k_exog > 0:
            self.ssm['state_intercept'] = np.zeros((self.k_states, self.nobs))
        idx = np.diag_indices(self.k_endog)
        self.ssm[('design',) + idx] = 1
        if self.k_ar > 0:
            idx = np.diag_indices((self.k_ar - 1) * self.k_endog)
            idx = (idx[0] + self.k_endog, idx[1])
            self.ssm[('transition',) + idx] = 1
        idx = np.diag_indices((self.k_ma - 1) * self.k_endog)
        idx = (idx[0] + (_min_k_ar + 1) * self.k_endog, idx[1] + _min_k_ar * self.k_endog)
        self.ssm[('transition',) + idx] = 1
        idx = np.diag_indices(self.k_endog)
        self.ssm[('selection',) + idx] = 1
        idx = (idx[0] + _min_k_ar * self.k_endog, idx[1])
        if self.k_ma > 0:
            self.ssm[('selection',) + idx] = 1
        if self._trend_is_const and self.k_exog == 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :]
        elif self.k_trend > 0 or self.k_exog > 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :-1]
        if self.k_ar > 0:
            self._idx_transition = np.s_['transition', :k_endog, :]
        else:
            self._idx_transition = np.s_['transition', :k_endog, k_endog:]
        if self.error_cov_type == 'diagonal':
            self._idx_state_cov = ('state_cov',) + np.diag_indices(self.k_endog)
        elif self.error_cov_type == 'unstructured':
            self._idx_lower_state_cov = np.tril_indices(self.k_endog)
        if self.measurement_error:
            self._idx_obs_cov = ('obs_cov',) + np.diag_indices(self.k_endog)

        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return (param_slice, offset)
        offset = 0
        self._params_trend, offset = _slice('trend', offset)
        self._params_ar, offset = _slice('ar', offset)
        self._params_ma, offset = _slice('ma', offset)
        self._params_regression, offset = _slice('regression', offset)
        self._params_state_cov, offset = _slice('state_cov', offset)
        self._params_obs_cov, offset = _slice('obs_cov', offset)
        self._final_exog = None
        self._init_keys += ['order', 'trend', 'error_cov_type', 'measurement_error', 'enforce_stationarity', 'enforce_invertibility'] + list(kwargs.keys())

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (VARMAXResults, VARMAXResultsWrapper)}

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)
        endog = pd.DataFrame(self.endog.copy())
        endog = endog.interpolate()
        endog = np.require(endog.bfill(), requirements='W')
        exog = None
        if self.k_trend > 0 and self.k_exog > 0:
            exog = np.c_[self._trend_data, self.exog]
        elif self.k_trend > 0:
            exog = self._trend_data
        elif self.k_exog > 0:
            exog = self.exog
        if np.any(np.isnan(endog)):
            mask = ~np.any(np.isnan(endog), axis=1)
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]
        trend_params = np.zeros(0)
        exog_params = np.zeros(0)
        if self.k_trend > 0 or self.k_exog > 0:
            trendexog_params = np.linalg.pinv(exog).dot(endog)
            endog -= np.dot(exog, trendexog_params)
            if self.k_trend > 0:
                trend_params = trendexog_params[:self.k_trend].T
            if self.k_endog > 0:
                exog_params = trendexog_params[self.k_trend:].T
        ar_params = []
        k_ar = self.k_ar if self.k_ar > 0 else 1
        mod_ar = var_model.VAR(endog)
        res_ar = mod_ar.fit(maxlags=k_ar, ic=None, trend='n')
        if self.k_ar > 0:
            ar_params = np.array(res_ar.params).T.ravel()
        endog = res_ar.resid
        if self.k_ar > 0 and self.enforce_stationarity:
            coefficient_matrices = ar_params.reshape(self.k_endog * self.k_ar, self.k_endog).T.reshape(self.k_endog, self.k_endog, self.k_ar).T
            stationary = is_invertible([1] + list(-coefficient_matrices))
            if not stationary:
                warn('Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.')
                ar_params *= 0
        ma_params = []
        if self.k_ma > 0:
            mod_ma = var_model.VAR(endog)
            res_ma = mod_ma.fit(maxlags=self.k_ma, ic=None, trend='n')
            ma_params = np.array(res_ma.params.T).ravel()
            if self.enforce_invertibility:
                coefficient_matrices = ma_params.reshape(self.k_endog * self.k_ma, self.k_endog).T.reshape(self.k_endog, self.k_endog, self.k_ma).T
                invertible = is_invertible([1] + list(-coefficient_matrices))
                if not invertible:
                    warn('Non-stationary starting moving-average parameters found. Using zeros as starting parameters.')
                    ma_params *= 0
        if self.k_ar > 0 and (self.k_trend > 0 or self.mle_regression):
            coefficient_matrices = ar_params.reshape(self.k_endog * self.k_ar, self.k_endog).T.reshape(self.k_endog, self.k_endog, self.k_ar).T
            tmp = np.eye(self.k_endog) - np.sum(coefficient_matrices, axis=0)
            if self.k_trend > 0:
                trend_params = np.dot(tmp, trend_params)
            if self.mle_regression > 0:
                exog_params = np.dot(tmp, exog_params)
        if self.k_trend > 0:
            params[self._params_trend] = trend_params.ravel()
        if self.k_ar > 0:
            params[self._params_ar] = ar_params
        if self.k_ma > 0:
            params[self._params_ma] = ma_params
        if self.mle_regression:
            params[self._params_regression] = exog_params.ravel()
        if self.error_cov_type == 'diagonal':
            params[self._params_state_cov] = res_ar.sigma_u.diagonal()
        elif self.error_cov_type == 'unstructured':
            cov_factor = np.linalg.cholesky(res_ar.sigma_u)
            params[self._params_state_cov] = cov_factor[self._idx_lower_state_cov].ravel()
        if self.measurement_error:
            if self.k_ma > 0:
                params[self._params_obs_cov] = res_ma.sigma_u.diagonal()
            else:
                params[self._params_obs_cov] = res_ar.sigma_u.diagonal()
        return params

    @property
    def param_names(self):
        param_names = []
        endog_names = self.endog_names
        if not isinstance(self.endog_names, list):
            endog_names = [endog_names]
        if self.k_trend > 0:
            for j in range(self.k_endog):
                for i in self.polynomial_trend.nonzero()[0]:
                    if i == 0:
                        param_names += ['intercept.%s' % endog_names[j]]
                    elif i == 1:
                        param_names += ['drift.%s' % endog_names[j]]
                    else:
                        param_names += ['trend.%d.%s' % (i, endog_names[j])]
        param_names += ['L%d.%s.%s' % (i + 1, endog_names[k], endog_names[j]) for j in range(self.k_endog) for i in range(self.k_ar) for k in range(self.k_endog)]
        param_names += ['L%d.e(%s).%s' % (i + 1, endog_names[k], endog_names[j]) for j in range(self.k_endog) for i in range(self.k_ma) for k in range(self.k_endog)]
        param_names += ['beta.{}.{}'.format(self.exog_names[j], endog_names[i]) for i in range(self.k_endog) for j in range(self.k_exog)]
        if self.error_cov_type == 'diagonal':
            param_names += ['sigma2.%s' % endog_names[i] for i in range(self.k_endog)]
        elif self.error_cov_type == 'unstructured':
            param_names += ['sqrt.var.%s' % endog_names[i] if i == j else 'sqrt.cov.{}.{}'.format(endog_names[j], endog_names[i]) for i in range(self.k_endog) for j in range(i + 1)]
        if self.measurement_error:
            param_names += ['measurement_variance.%s' % endog_names[i] for i in range(self.k_endog)]
        return param_names

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
        Constrains the factor transition to be stationary and variances to be
        positive.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)
        constrained[self._params_trend] = unconstrained[self._params_trend]
        if self.k_ar > 0 and self.enforce_stationarity:
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(unconstrained[self._params_state_cov] ** 2)
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape, dtype=unconstrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = unconstrained[self._params_state_cov]
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)
            coefficients = unconstrained[self._params_ar].reshape(self.k_endog, self.k_endog * self.k_ar)
            coefficient_matrices, variance = constrain_stationary_multivariate(coefficients, state_cov)
            constrained[self._params_ar] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ar] = unconstrained[self._params_ar]
        if self.k_ma > 0 and self.enforce_invertibility:
            state_cov = np.eye(self.k_endog, dtype=unconstrained.dtype)
            coefficients = unconstrained[self._params_ma].reshape(self.k_endog, self.k_endog * self.k_ma)
            coefficient_matrices, variance = constrain_stationary_multivariate(coefficients, state_cov)
            constrained[self._params_ma] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ma] = unconstrained[self._params_ma]
        constrained[self._params_regression] = unconstrained[self._params_regression]
        if self.error_cov_type == 'diagonal':
            constrained[self._params_state_cov] = unconstrained[self._params_state_cov] ** 2
        elif self.error_cov_type == 'unstructured':
            constrained[self._params_state_cov] = unconstrained[self._params_state_cov]
        if self.measurement_error:
            constrained[self._params_obs_cov] = unconstrained[self._params_obs_cov] ** 2
        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, dtype=constrained.dtype)
        unconstrained[self._params_trend] = constrained[self._params_trend]
        if self.k_ar > 0 and self.enforce_stationarity:
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(constrained[self._params_state_cov])
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape, dtype=constrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = constrained[self._params_state_cov]
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)
            coefficients = constrained[self._params_ar].reshape(self.k_endog, self.k_endog * self.k_ar)
            unconstrained_matrices, variance = unconstrain_stationary_multivariate(coefficients, state_cov)
            unconstrained[self._params_ar] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ar] = constrained[self._params_ar]
        if self.k_ma > 0 and self.enforce_invertibility:
            state_cov = np.eye(self.k_endog, dtype=constrained.dtype)
            coefficients = constrained[self._params_ma].reshape(self.k_endog, self.k_endog * self.k_ma)
            unconstrained_matrices, variance = unconstrain_stationary_multivariate(coefficients, state_cov)
            unconstrained[self._params_ma] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ma] = constrained[self._params_ma]
        unconstrained[self._params_regression] = constrained[self._params_regression]
        if self.error_cov_type == 'diagonal':
            unconstrained[self._params_state_cov] = constrained[self._params_state_cov] ** 0.5
        elif self.error_cov_type == 'unstructured':
            unconstrained[self._params_state_cov] = constrained[self._params_state_cov]
        if self.measurement_error:
            unconstrained[self._params_obs_cov] = constrained[self._params_obs_cov] ** 0.5
        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super()._validate_can_fix_params(param_names)
        ix = np.cumsum(list(self.parameters.values()))[:-1]
        _, ar_names, ma_names, _, _, _ = (arr.tolist() for arr in np.array_split(self.param_names, ix))
        if self.enforce_stationarity and self.k_ar > 0:
            if self.k_endog > 1 or self.k_ar > 1:
                fix_all = param_names.issuperset(ar_names)
                fix_any = len(param_names.intersection(ar_names)) > 0
                if fix_any and (not fix_all):
                    raise ValueError('Cannot fix individual autoregressive parameters when `enforce_stationarity=True`. In this case, must either fix all autoregressive parameters or none.')
        if self.enforce_invertibility and self.k_ma > 0:
            if self.k_endog or self.k_ma > 1:
                fix_all = param_names.issuperset(ma_names)
                fix_any = len(param_names.intersection(ma_names)) > 0
                if fix_any and (not fix_all):
                    raise ValueError('Cannot fix individual moving average parameters when `enforce_invertibility=True`. In this case, must either fix all moving average parameters or none.')

    def update(self, params, transformed=True, includes_fixed=False, complex_step=False):
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        if self.mle_regression:
            exog_params = params[self._params_regression].reshape(self.k_endog, self.k_exog).T
            intercept = np.dot(self.exog[1:], exog_params)
            self.ssm[self._idx_state_intercept] = intercept.T
            if self._final_exog is not None:
                self.ssm['state_intercept', :self.k_endog, -1] = np.dot(self._final_exog, exog_params)
        if self.k_trend > 0:
            if not self.mle_regression:
                zero = np.array(0, dtype=params.dtype)
                self.ssm['state_intercept', :] = zero
            trend_params = params[self._params_trend].reshape(self.k_endog, self.k_trend).T
            if self._trend_is_const:
                intercept = trend_params
            else:
                intercept = np.dot(self._trend_data[1:], trend_params)
            self.ssm[self._idx_state_intercept] += intercept.T
            if self._final_trend is not None and self._idx_state_intercept[-1].stop == -1:
                self.ssm['state_intercept', :self.k_endog, -1:] += np.dot(self._final_trend, trend_params).T
        if self.mle_regression and self._final_exog is None:
            nan = np.array(np.nan, dtype=params.dtype)
            self.ssm['state_intercept', :self.k_endog, -1] = nan
        ar = params[self._params_ar].reshape(self.k_endog, self.k_endog * self.k_ar)
        ma = params[self._params_ma].reshape(self.k_endog, self.k_endog * self.k_ma)
        self.ssm[self._idx_transition] = np.c_[ar, ma]
        if self.error_cov_type == 'diagonal':
            self.ssm[self._idx_state_cov] = params[self._params_state_cov]
        elif self.error_cov_type == 'unstructured':
            state_cov_lower = np.zeros(self.ssm['state_cov'].shape, dtype=params.dtype)
            state_cov_lower[self._idx_lower_state_cov] = params[self._params_state_cov]
            self.ssm['state_cov'] = np.dot(state_cov_lower, state_cov_lower.T)
        if self.measurement_error:
            self.ssm[self._idx_obs_cov] = params[self._params_obs_cov]

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for simulating or forecasting with `exog` or
        trend, because if we had these then the last predicted_state has been
        set to NaN since we did not have the appropriate `exog` to create it.
        Since we handle trend in the same way as `exog`, we still have this
        issue when only trend is used without `exog`.
        """
        cache_value = self._final_exog
        if self.k_exog > 0:
            if exog is not None:
                exog = np.atleast_1d(exog)
                if exog.ndim == 2:
                    exog = exog[:1]
                try:
                    exog = np.reshape(exog[:1], (self.k_exog,))
                except ValueError:
                    raise ValueError('Provided exogenous values are not of the appropriate shape. Required %s, got %s.' % (str((self.k_exog,)), str(exog.shape)))
            self._final_exog = exog
        try:
            yield
        finally:
            self._final_exog = cache_value

    @Appender(MLEModel.simulate.__doc__)
    def simulate(self, params, nsimulations, measurement_shocks=None, state_shocks=None, initial_state=None, anchor=None, repetitions=None, exog=None, extend_model=None, extend_kwargs=None, transformed=True, includes_fixed=False, **kwargs):
        with self._set_final_exog(exog):
            out = super().simulate(params, nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state, anchor=anchor, repetitions=repetitions, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, transformed=transformed, includes_fixed=includes_fixed, **kwargs)
        return out