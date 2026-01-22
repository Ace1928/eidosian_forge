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
        This context manager calls the model-level context manager and
        additionally updates the last element of filter_results.state_intercept
        appropriately.
        """
    mod = self.model
    with mod._set_final_exog(exog):
        cache_value = self.filter_results.state_intercept[:, -1]
        mod.update(self.params)
        self.filter_results.state_intercept[:mod.k_endog, -1] = mod['state_intercept', :mod.k_endog, -1]
        try:
            yield
        finally:
            self.filter_results.state_intercept[:, -1] = cache_value