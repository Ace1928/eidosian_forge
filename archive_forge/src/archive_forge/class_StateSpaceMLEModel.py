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
class StateSpaceMLEModel(tsbase.TimeSeriesModel):
    """
    This is a temporary base model from ETS, here I just copy everything I need
    from statespace.mlemodel.MLEModel
    """

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none', **kwargs):
        super().__init__(endog=endog, exog=exog, dates=dates, freq=freq, missing=missing)
        self._init_kwargs = kwargs
        self.endog, self.exog = self.prepare_data(self.data)
        self.use_pandas = isinstance(self.data, PandasData)
        self.nobs = self.endog.shape[0]
        self._has_fixed_params = False
        self._fixed_params = None
        self._params_index = None
        self._fixed_params_index = None
        self._free_params_index = None

    @staticmethod
    def prepare_data(data):
        raise NotImplementedError

    def clone(self, endog, exog=None, **kwargs):
        raise NotImplementedError

    def _validate_can_fix_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError('Invalid parameter name passed: "%s".' % param_name)

    @property
    def k_params(self):
        return len(self.param_names)

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
        if self._fixed_params is None:
            self._fixed_params = {}
            self._params_index = OrderedDict(zip(self.param_names, np.arange(self.k_params)))
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index
        all_fixed_param_names = set(params.keys()) | set(self._fixed_params.keys())
        self._validate_can_fix_params(all_fixed_param_names)
        self._fixed_params.update(params)
        self._fixed_params = OrderedDict([(name, self._fixed_params[name]) for name in self.param_names if name in self._fixed_params])
        self._has_fixed_params = True
        self._fixed_params_index = [self._params_index[key] for key in self._fixed_params.keys()]
        self._free_params_index = list(set(np.arange(self.k_params)).difference(self._fixed_params_index))
        try:
            yield
        finally:
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

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

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        """
        Not implemented for state space models
        """
        raise NotImplementedError

    def _wrap_data(self, data, start_idx, end_idx, names=None):
        if data.ndim > 1 and data.shape[1] == 1:
            data = np.squeeze(data, axis=1)
        if self.use_pandas:
            if data.shape[0]:
                _, _, _, index = self._get_prediction_index(start_idx, end_idx)
            else:
                index = None
            if data.ndim < 2:
                data = pd.Series(data, index=index, name=names)
            else:
                data = pd.DataFrame(data, index=index, columns=names)
        return data

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

    def _score_complex_step(self, params, **kwargs):
        epsilon = _get_epsilon(params, 2.0, None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        return approx_fprime_cs(params, self.loglike, epsilon=epsilon, kwargs=kwargs)

    def _score_finite_difference(self, params, approx_centered=False, **kwargs):
        kwargs['transformed'] = True
        return approx_fprime(params, self.loglike, kwargs=kwargs, centered=approx_centered)

    def _hessian_finite_difference(self, params, approx_centered=False, **kwargs):
        params = np.array(params, ndmin=1)
        warnings.warn('Calculation of the Hessian using finite differences is usually subject to substantial approximation errors.', PrecisionWarning, stacklevel=3)
        if not approx_centered:
            epsilon = _get_epsilon(params, 3, None, len(params))
        else:
            epsilon = _get_epsilon(params, 4, None, len(params)) / 2
        hessian = approx_fprime(params, self._score_finite_difference, epsilon=epsilon, kwargs=kwargs, centered=approx_centered)
        return hessian / self.nobs_effective

    def _hessian_complex_step(self, params, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
        epsilon = _get_epsilon(params, 3.0, None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        hessian = approx_hess_cs(params, self.loglike, epsilon=epsilon, kwargs=kwargs)
        return hessian / self.nobs_effective