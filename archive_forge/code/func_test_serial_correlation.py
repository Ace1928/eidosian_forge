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
def test_serial_correlation(self, method, df_adjust=False, lags=None):
    """
        Ljung-Box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        method : {'ljungbox','boxpierece', None}
            The statistical test for serial correlation. If None, an attempt is
            made to select an appropriate test.
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is min(10, nobs // 5) for
            non-seasonal models and min(2*m, nobs // 5) for seasonal time
            series where m is the seasonal period.
        df_adjust : bool, optional
            If True, the degrees of freedom consumed by the model is subtracted
            from the degrees-of-freedom used in the test so that the adjusted
            dof for the statistics are lags - model_df. In an ARMA model, this
            value is usually p+q where p is the AR order and q is the MA order.
            When using df_adjust, it is not possible to use tests based on
            fewer than model_df lags.
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
        Let `d` = max(loglikelihood_burn, nobs_diffuse); this test is
        calculated ignoring the first `d` residuals.

        Output is nan for any endogenous variable which has missing values.
        """
    if method is None:
        method = 'ljungbox'
    if self.standardized_forecasts_error is None:
        raise ValueError('Cannot compute test statistic when standardized forecast errors have not been computed.')
    if method == 'ljungbox' or method == 'boxpierce':
        from statsmodels.stats.diagnostic import acorr_ljungbox
        d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
        nobs_effective = self.nobs - d
        output = []
        if lags is None:
            seasonal_periods = getattr(self.model, 'seasonal_periods', 0)
            if seasonal_periods:
                lags = min(2 * seasonal_periods, nobs_effective // 5)
            else:
                lags = min(10, nobs_effective // 5)
        model_df = 0
        if df_adjust:
            model_df = max(0, self.df_model - self.k_diffuse_states - 1)
        cols = [2, 3] if method == 'boxpierce' else [0, 1]
        for i in range(self.model.k_endog):
            results = acorr_ljungbox(self.filter_results.standardized_forecasts_error[i][d:], lags=lags, boxpierce=method == 'boxpierce', model_df=model_df)
            output.append(np.asarray(results)[:, cols].T)
        output = np.c_[output]
    else:
        raise NotImplementedError('Invalid serial correlation test method.')
    return output