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
        