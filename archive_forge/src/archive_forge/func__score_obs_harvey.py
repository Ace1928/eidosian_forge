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