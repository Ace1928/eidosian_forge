import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender
class DynamicFactor(MLEModel):
    """
    Dynamic factor model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the observation error term,
        where "unstructured" puts no restrictions on the matrix, "diagonal"
        requires it to be any diagonal matrix (uncorrelated errors), and
        "scalar" requires it to be a scalar times the identity matrix. Default
        is "diagonal".
    error_order : int, optional
        The order of the vector autoregression followed by the observation
        error component. Default is None, corresponding to white noise errors.
    error_var : bool, optional
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'diagonal', 'unstructured'}
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors).
    error_order : int
        The order of the vector autoregression followed by the observation
        error component.
    error_var : bool
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.

    Notes
    -----
    The dynamic factor model considered here is in the so-called static form,
    and is specified:

    .. math::

        y_t & = \\Lambda f_t + B x_t + u_t \\\\
        f_t & = A_1 f_{t-1} + \\dots + A_p f_{t-p} + \\eta_t \\\\
        u_t & = C_1 u_{t-1} + \\dots + C_q u_{t-q} + \\varepsilon_t

    where there are `k_endog` observed series and `k_factors` unobserved
    factors. Thus :math:`y_t` is a `k_endog` x 1 vector and :math:`f_t` is a
    `k_factors` x 1 vector.

    :math:`x_t` are optional exogenous vectors, shaped `k_exog` x 1.

    :math:`\\eta_t` and :math:`\\varepsilon_t` are white noise error terms. In
    order to identify the factors, :math:`Var(\\eta_t) = I`. Denote
    :math:`Var(\\varepsilon_t) \\equiv \\Sigma`.

    Options related to the unobserved factors:

    - `k_factors`: this is the dimension of the vector :math:`f_t`, above.
      To exclude factors completely, set `k_factors = 0`.
    - `factor_order`: this is the number of lags to include in the factor
      evolution equation, and corresponds to :math:`p`, above. To have static
      factors, set `factor_order = 0`.

    Options related to the observation error term :math:`u_t`:

    - `error_order`: the number of lags to include in the error evolution
      equation; corresponds to :math:`q`, above. To have white noise errors,
      set `error_order = 0` (this is the default).
    - `error_cov_type`: this controls the form of the covariance matrix
      :math:`\\Sigma`. If it is "dscalar", then :math:`\\Sigma = \\sigma^2 I`. If
      it is "diagonal", then
      :math:`\\Sigma = \\text{diag}(\\sigma_1^2, \\dots, \\sigma_n^2)`. If it is
      "unstructured", then :math:`\\Sigma` is any valid variance / covariance
      matrix (i.e. symmetric and positive definite).
    - `error_var`: this controls whether or not the errors evolve jointly
      according to a VAR(q), or individually according to separate AR(q)
      processes. In terms of the formulation above, if `error_var = False`,
      then the matrices :math:C_i` are diagonal, otherwise they are general
      VAR matrices.

    References
    ----------
    .. [*] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, k_factors, factor_order, exog=None, error_order=0, error_var=False, error_cov_type='diagonal', enforce_stationarity=True, **kwargs):
        self.enforce_stationarity = enforce_stationarity
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var and error_order > 0
        self.error_cov_type = error_cov_type
        self.k_exog, exog = prepare_exog(exog)
        self.mle_regression = self.k_exog > 0
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog, order='C')
        k_endog = endog.shape[1] if endog.ndim > 1 else 1
        self._factor_order = max(1, self.factor_order) * self.k_factors
        self._error_order = self.error_order * k_endog
        k_states = self._factor_order
        k_posdef = self.k_factors
        if self.error_order > 0:
            k_states += self._error_order
            k_posdef += k_endog
        self._unused_state = False
        if k_states == 0:
            k_states = 1
            k_posdef = 1
            self._unused_state = True
        if k_endog < 2:
            raise ValueError('The dynamic factors model is only valid for multivariate time series.')
        if self.k_factors >= k_endog:
            raise ValueError('Number of factors must be less than the number of endogenous variables.')
        if self.error_cov_type not in ['scalar', 'diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type specification.')
        kwargs.setdefault('initialization', 'stationary')
        super().__init__(endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        if self.k_exog > 0:
            self.ssm._time_invariant = False
        self.parameters = {}
        self._initialize_loadings()
        self._initialize_exog()
        self._initialize_error_cov()
        self._initialize_factor_transition()
        self._initialize_error_transition()
        self.k_params = sum(self.parameters.values())

        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return (param_slice, offset)
        offset = 0
        self._params_loadings, offset = _slice('factor_loadings', offset)
        self._params_exog, offset = _slice('exog', offset)
        self._params_error_cov, offset = _slice('error_cov', offset)
        self._params_factor_transition, offset = _slice('factor_transition', offset)
        self._params_error_transition, offset = _slice('error_transition', offset)
        self._init_keys += ['k_factors', 'factor_order', 'error_order', 'error_var', 'error_cov_type', 'enforce_stationarity'] + list(kwargs.keys())

    def _initialize_loadings(self):
        self.parameters['factor_loadings'] = self.k_endog * self.k_factors
        if self.error_order > 0:
            start = self._factor_order
            end = self._factor_order + self.k_endog
            self.ssm['design', :, start:end] = np.eye(self.k_endog)
        self._idx_loadings = np.s_['design', :, :self.k_factors]

    def _initialize_exog(self):
        self.parameters['exog'] = self.k_exog * self.k_endog
        if self.k_exog > 0:
            self.ssm['obs_intercept'] = np.zeros((self.k_endog, self.nobs))
        self._idx_exog = np.s_['obs_intercept', :self.k_endog, :]

    def _initialize_error_cov(self):
        if self.error_cov_type == 'scalar':
            self._initialize_error_cov_diagonal(scalar=True)
        elif self.error_cov_type == 'diagonal':
            self._initialize_error_cov_diagonal(scalar=False)
        elif self.error_cov_type == 'unstructured':
            self._initialize_error_cov_unstructured()

    def _initialize_error_cov_diagonal(self, scalar=False):
        self.parameters['error_cov'] = 1 if scalar else self.k_endog
        k_endog = self.k_endog
        k_factors = self.k_factors
        idx = np.diag_indices(k_endog)
        if self.error_order > 0:
            matrix = 'state_cov'
            idx = (idx[0] + k_factors, idx[1] + k_factors)
        else:
            matrix = 'obs_cov'
        self._idx_error_cov = (matrix,) + idx

    def _initialize_error_cov_unstructured(self):
        k_endog = self.k_endog
        self.parameters['error_cov'] = int(k_endog * (k_endog + 1) / 2)
        self._idx_lower_error_cov = np.tril_indices(self.k_endog)
        if self.error_order > 0:
            start = self.k_factors
            end = self.k_factors + self.k_endog
            self._idx_error_cov = np.s_['state_cov', start:end, start:end]
        else:
            self._idx_error_cov = np.s_['obs_cov', :, :]

    def _initialize_factor_transition(self):
        order = self.factor_order * self.k_factors
        k_factors = self.k_factors
        self.parameters['factor_transition'] = self.factor_order * self.k_factors ** 2
        if self.k_factors > 0:
            if self.factor_order > 0:
                self.ssm['transition', k_factors:order, :order - k_factors] = np.eye(order - k_factors)
            self.ssm['selection', :k_factors, :k_factors] = np.eye(k_factors)
            self.ssm['state_cov', :k_factors, :k_factors] = np.eye(k_factors)
        self._idx_factor_transition = np.s_['transition', :k_factors, :order]

    def _initialize_error_transition(self):
        if self.error_order == 0:
            self._initialize_error_transition_white_noise()
        else:
            k_endog = self.k_endog
            k_factors = self.k_factors
            _factor_order = self._factor_order
            _error_order = self._error_order
            _slice = np.s_['selection', _factor_order:_factor_order + k_endog, k_factors:k_factors + k_endog]
            self.ssm[_slice] = np.eye(k_endog)
            _slice = np.s_['transition', _factor_order + k_endog:_factor_order + _error_order, _factor_order:_factor_order + _error_order - k_endog]
            self.ssm[_slice] = np.eye(_error_order - k_endog)
            if self.error_var:
                self._initialize_error_transition_var()
            else:
                self._initialize_error_transition_individual()

    def _initialize_error_transition_white_noise(self):
        self.parameters['error_transition'] = 0
        self._idx_error_transition = np.s_['transition', 0:0, 0:0]

    def _initialize_error_transition_var(self):
        k_endog = self.k_endog
        _factor_order = self._factor_order
        _error_order = self._error_order
        self.parameters['error_transition'] = _error_order * k_endog
        self._idx_error_transition = np.s_['transition', _factor_order:_factor_order + k_endog, _factor_order:_factor_order + _error_order]

    def _initialize_error_transition_individual(self):
        k_endog = self.k_endog
        _error_order = self._error_order
        self.parameters['error_transition'] = _error_order
        idx = np.tile(np.diag_indices(k_endog), self.error_order)
        row_shift = self._factor_order
        col_inc = self._factor_order + np.repeat([i * k_endog for i in range(self.error_order)], k_endog)
        idx[0] += row_shift
        idx[1] += col_inc
        idx_diag = idx.copy()
        idx_diag[0] -= row_shift
        idx_diag[1] -= self._factor_order
        idx_diag = idx_diag[:, np.lexsort((idx_diag[1], idx_diag[0]))]
        self._idx_error_diag = (idx_diag[0], idx_diag[1])
        idx = idx[:, np.lexsort((idx[1], idx[0]))]
        self._idx_error_transition = np.s_['transition', idx[0], idx[1]]

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (DynamicFactorResults, DynamicFactorResultsWrapper)}

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)
        endog = self.endog.copy()
        mask = ~np.any(np.isnan(endog), axis=1)
        endog = endog[mask]
        if self.k_exog > 0:
            exog = self.exog[mask]
        if self.k_factors > 0:
            res_pca = PCA(endog, ncomp=self.k_factors)
            mod_ols = OLS(endog, res_pca.factors)
            res_ols = mod_ols.fit()
            params[self._params_loadings] = res_ols.params.T.ravel()
            endog = endog - np.dot(res_pca.factors, res_pca.loadings.T)
        if self.k_exog > 0:
            mod_ols = OLS(endog, exog=exog)
            res_ols = mod_ols.fit()
            params[self._params_exog] = res_ols.params.T.ravel()
            endog = res_ols.resid
        stationary = True
        if self.k_factors > 1 and self.factor_order > 0:
            mod_factors = VAR(res_pca.factors)
            res_factors = mod_factors.fit(maxlags=self.factor_order, ic=None, trend='n')
            params[self._params_factor_transition] = res_factors.params.T.ravel()
            coefficient_matrices = params[self._params_factor_transition].reshape(self.k_factors * self.factor_order, self.k_factors).T.reshape(self.k_factors, self.k_factors, self.factor_order).T
            stationary = is_invertible([1] + list(-coefficient_matrices))
        elif self.k_factors > 0 and self.factor_order > 0:
            Y = res_pca.factors[self.factor_order:]
            X = lagmat(res_pca.factors, self.factor_order, trim='both')
            params_ar = np.linalg.pinv(X).dot(Y)
            stationary = is_invertible(np.r_[1, -params_ar.squeeze()])
            params[self._params_factor_transition] = params_ar[:, 0]
        if not stationary and self.enforce_stationarity:
            raise ValueError('Non-stationary starting autoregressive parameters found with `enforce_stationarity` set to True.')
        if self.error_order == 0:
            if self.error_cov_type == 'scalar':
                params[self._params_error_cov] = endog.var(axis=0).mean()
            elif self.error_cov_type == 'diagonal':
                params[self._params_error_cov] = endog.var(axis=0)
            elif self.error_cov_type == 'unstructured':
                cov_factor = np.diag(endog.std(axis=0))
                params[self._params_error_cov] = cov_factor[self._idx_lower_error_cov].ravel()
        elif self.error_var:
            mod_errors = VAR(endog)
            res_errors = mod_errors.fit(maxlags=self.error_order, ic=None, trend='n')
            coefficient_matrices = np.array(res_errors.params.T).ravel().reshape(self.k_endog * self.error_order, self.k_endog).T.reshape(self.k_endog, self.k_endog, self.error_order).T
            stationary = is_invertible([1] + list(-coefficient_matrices))
            if not stationary and self.enforce_stationarity:
                raise ValueError('Non-stationary starting error autoregressive parameters found with `enforce_stationarity` set to True.')
            params[self._params_error_transition] = np.array(res_errors.params.T).ravel()
            if self.error_cov_type == 'scalar':
                params[self._params_error_cov] = res_errors.sigma_u.diagonal().mean()
            elif self.error_cov_type == 'diagonal':
                params[self._params_error_cov] = res_errors.sigma_u.diagonal()
            elif self.error_cov_type == 'unstructured':
                try:
                    cov_factor = np.linalg.cholesky(res_errors.sigma_u)
                except np.linalg.LinAlgError:
                    cov_factor = np.eye(res_errors.sigma_u.shape[0]) * res_errors.sigma_u.diagonal().mean() ** 0.5
                cov_factor = np.eye(res_errors.sigma_u.shape[0]) * res_errors.sigma_u.diagonal().mean() ** 0.5
                params[self._params_error_cov] = cov_factor[self._idx_lower_error_cov].ravel()
        else:
            error_ar_params = []
            error_cov_params = []
            for i in range(self.k_endog):
                mod_error = ARIMA(endog[:, i], order=(self.error_order, 0, 0), trend='n', enforce_stationarity=True)
                res_error = mod_error.fit(method='burg')
                error_ar_params += res_error.params[:self.error_order].tolist()
                error_cov_params += res_error.params[-1:].tolist()
            params[self._params_error_transition] = np.r_[error_ar_params]
            params[self._params_error_cov] = np.r_[error_cov_params]
        return params

    @property
    def param_names(self):
        param_names = []
        endog_names = self.endog_names
        param_names += ['loading.f%d.%s' % (j + 1, endog_names[i]) for i in range(self.k_endog) for j in range(self.k_factors)]
        param_names += ['beta.{}.{}'.format(self.exog_names[j], endog_names[i]) for i in range(self.k_endog) for j in range(self.k_exog)]
        if self.error_cov_type == 'scalar':
            param_names += ['sigma2']
        elif self.error_cov_type == 'diagonal':
            param_names += ['sigma2.%s' % endog_names[i] for i in range(self.k_endog)]
        elif self.error_cov_type == 'unstructured':
            param_names += ['cov.chol[%d,%d]' % (i + 1, j + 1) for i in range(self.k_endog) for j in range(i + 1)]
        param_names += ['L%d.f%d.f%d' % (i + 1, k + 1, j + 1) for j in range(self.k_factors) for i in range(self.factor_order) for k in range(self.k_factors)]
        if self.error_var:
            param_names += ['L%d.e(%s).e(%s)' % (i + 1, endog_names[k], endog_names[j]) for j in range(self.k_endog) for i in range(self.error_order) for k in range(self.k_endog)]
        else:
            param_names += ['L%d.e(%s).e(%s)' % (i + 1, endog_names[j], endog_names[j]) for j in range(self.k_endog) for i in range(self.error_order)]
        return param_names

    @property
    def state_names(self):
        names = []
        endog_names = self.endog_names
        names += ['f%d' % (j + 1) if i == 0 else 'f%d.L%d' % (j + 1, i) for i in range(max(1, self.factor_order)) for j in range(self.k_factors)]
        if self.error_order > 0:
            names += ['e(%s)' % endog_names[j] if i == 0 else 'e(%s).L%d' % (endog_names[j], i) for i in range(self.error_order) for j in range(self.k_endog)]
        if self._unused_state:
            names += ['dummy']
        return names

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
        dtype = unconstrained.dtype
        constrained = np.zeros(unconstrained.shape, dtype=dtype)
        constrained[self._params_loadings] = unconstrained[self._params_loadings]
        constrained[self._params_exog] = unconstrained[self._params_exog]
        if self.error_cov_type in ['scalar', 'diagonal']:
            constrained[self._params_error_cov] = unconstrained[self._params_error_cov] ** 2
        elif self.error_cov_type == 'unstructured':
            constrained[self._params_error_cov] = unconstrained[self._params_error_cov]
        if self.enforce_stationarity and self.factor_order > 0:
            unconstrained_matrices = unconstrained[self._params_factor_transition].reshape(self.k_factors, self._factor_order)
            cov = self.ssm['state_cov', :self.k_factors, :self.k_factors].real
            coefficient_matrices, variance = constrain_stationary_multivariate(unconstrained_matrices, cov)
            constrained[self._params_factor_transition] = coefficient_matrices.ravel()
        else:
            constrained[self._params_factor_transition] = unconstrained[self._params_factor_transition]
        if self.enforce_stationarity and self.error_order > 0:
            if self.error_var:
                unconstrained_matrices = unconstrained[self._params_error_transition].reshape(self.k_endog, self._error_order)
                start = self.k_factors
                end = self.k_factors + self.k_endog
                cov = self.ssm['state_cov', start:end, start:end].real
                coefficient_matrices, variance = constrain_stationary_multivariate(unconstrained_matrices, cov)
                constrained[self._params_error_transition] = coefficient_matrices.ravel()
            else:
                coefficients = unconstrained[self._params_error_transition].copy()
                for i in range(self.k_endog):
                    start = i * self.error_order
                    end = (i + 1) * self.error_order
                    coefficients[start:end] = constrain_stationary_univariate(coefficients[start:end])
                constrained[self._params_error_transition] = coefficients
        else:
            constrained[self._params_error_transition] = unconstrained[self._params_error_transition]
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
        dtype = constrained.dtype
        unconstrained = np.zeros(constrained.shape, dtype=dtype)
        unconstrained[self._params_loadings] = constrained[self._params_loadings]
        unconstrained[self._params_exog] = constrained[self._params_exog]
        if self.error_cov_type in ['scalar', 'diagonal']:
            unconstrained[self._params_error_cov] = constrained[self._params_error_cov] ** 0.5
        elif self.error_cov_type == 'unstructured':
            unconstrained[self._params_error_cov] = constrained[self._params_error_cov]
        if self.enforce_stationarity and self.factor_order > 0:
            constrained_matrices = constrained[self._params_factor_transition].reshape(self.k_factors, self._factor_order)
            cov = self.ssm['state_cov', :self.k_factors, :self.k_factors].real
            coefficient_matrices, variance = unconstrain_stationary_multivariate(constrained_matrices, cov)
            unconstrained[self._params_factor_transition] = coefficient_matrices.ravel()
        else:
            unconstrained[self._params_factor_transition] = constrained[self._params_factor_transition]
        if self.enforce_stationarity and self.error_order > 0:
            if self.error_var:
                constrained_matrices = constrained[self._params_error_transition].reshape(self.k_endog, self._error_order)
                start = self.k_factors
                end = self.k_factors + self.k_endog
                cov = self.ssm['state_cov', start:end, start:end].real
                coefficient_matrices, variance = unconstrain_stationary_multivariate(constrained_matrices, cov)
                unconstrained[self._params_error_transition] = coefficient_matrices.ravel()
            else:
                coefficients = constrained[self._params_error_transition].copy()
                for i in range(self.k_endog):
                    start = i * self.error_order
                    end = (i + 1) * self.error_order
                    coefficients[start:end] = unconstrain_stationary_univariate(coefficients[start:end])
                unconstrained[self._params_error_transition] = coefficients
        else:
            unconstrained[self._params_error_transition] = constrained[self._params_error_transition]
        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super()._validate_can_fix_params(param_names)
        ix = np.cumsum(list(self.parameters.values()))[:-1]
        _, _, _, factor_transition_names, error_transition_names = (arr.tolist() for arr in np.array_split(self.param_names, ix))
        if self.enforce_stationarity and self.factor_order > 0:
            if self.k_factors > 1 or self.factor_order > 1:
                fix_all = param_names.issuperset(factor_transition_names)
                fix_any = len(param_names.intersection(factor_transition_names)) > 0
                if fix_any and (not fix_all):
                    raise ValueError('Cannot fix individual factor transition parameters when `enforce_stationarity=True`. In this case, must either fix all factor transition parameters or none.')
        if self.enforce_stationarity and self.error_order > 0:
            if self.error_var or self.error_order > 1:
                fix_all = param_names.issuperset(error_transition_names)
                fix_any = len(param_names.intersection(error_transition_names)) > 0
                if fix_any and (not fix_all):
                    raise ValueError('Cannot fix individual error transition parameters when `enforce_stationarity=True`. In this case, must either fix all error transition parameters or none.')

    def update(self, params, transformed=True, includes_fixed=False, complex_step=False):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Let `n = k_endog`, `m = k_factors`, and `p = factor_order`. Then the
        `params` vector has length
        :math:`[n 	imes m] + [n] + [m^2 	imes p]`.
        It is expanded in the following way:

        - The first :math:`n 	imes m` parameters fill out the factor loading
          matrix, starting from the [0,0] entry and then proceeding along rows.
          These parameters are not modified in `transform_params`.
        - The next :math:`n` parameters provide variances for the error_cov
          errors in the observation equation. They fill in the diagonal of the
          observation covariance matrix, and are constrained to be positive by
          `transofrm_params`.
        - The next :math:`m^2 	imes p` parameters are used to create the `p`
          coefficient matrices for the vector autoregression describing the
          factor transition. They are transformed in `transform_params` to
          enforce stationarity of the VAR(p). They are placed so as to make
          the transition matrix a companion matrix for the VAR. In particular,
          we assume that the first :math:`m^2` parameters fill the first
          coefficient matrix (starting at [0,0] and filling along rows), the
          second :math:`m^2` parameters fill the second matrix, etc.
        """
        params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
        self.ssm[self._idx_loadings] = params[self._params_loadings].reshape(self.k_endog, self.k_factors)
        if self.k_exog > 0:
            exog_params = params[self._params_exog].reshape(self.k_endog, self.k_exog).T
            self.ssm[self._idx_exog] = np.dot(self.exog, exog_params).T
        if self.error_cov_type in ['scalar', 'diagonal']:
            self.ssm[self._idx_error_cov] = params[self._params_error_cov]
        elif self.error_cov_type == 'unstructured':
            error_cov_lower = np.zeros((self.k_endog, self.k_endog), dtype=params.dtype)
            error_cov_lower[self._idx_lower_error_cov] = params[self._params_error_cov]
            self.ssm[self._idx_error_cov] = np.dot(error_cov_lower, error_cov_lower.T)
        self.ssm[self._idx_factor_transition] = params[self._params_factor_transition].reshape(self.k_factors, self.factor_order * self.k_factors)
        if self.error_var:
            self.ssm[self._idx_error_transition] = params[self._params_error_transition].reshape(self.k_endog, self._error_order)
        else:
            self.ssm[self._idx_error_transition] = params[self._params_error_transition]