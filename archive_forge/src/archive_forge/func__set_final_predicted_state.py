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
@contextlib.contextmanager
def _set_final_predicted_state(self, exog, out_of_sample):
    """
        Set the final predicted state value using out-of-sample `exog` / trend

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
        We need special handling for forecasting with `exog`, because
        if we had these then the last predicted_state has been set to NaN since
        we did not have the appropriate `exog` to create it.
        """
    flag = out_of_sample and self.model.k_exog > 0
    if flag:
        tmp_endog = concat([self.model.endog[-1:], np.zeros((1, self.model.k_endog))])
        if self.model.k_exog > 0:
            tmp_exog = concat([self.model.exog[-1:], exog[:1]])
        else:
            tmp_exog = None
        tmp_trend_offset = self.model.trend_offset + self.nobs - 1
        tmp_mod = self.model.clone(tmp_endog, exog=tmp_exog, trend_offset=tmp_trend_offset)
        constant = self.filter_results.predicted_state[:, -2]
        stationary_cov = self.filter_results.predicted_state_cov[:, :, -2]
        tmp_mod.ssm.initialize_known(constant=constant, stationary_cov=stationary_cov)
        tmp_res = tmp_mod.filter(self.params, transformed=True, includes_fixed=True, return_ssm=True)
        self.filter_results.predicted_state[:, -1] = tmp_res.predicted_state[:, -2]
    try:
        yield
    finally:
        if flag:
            self.filter_results.predicted_state[:, -1] = np.nan