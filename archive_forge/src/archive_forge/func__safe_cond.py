import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _safe_cond(a):
    """Compute condition while protecting from LinAlgError"""
    try:
        return np.linalg.cond(a)
    except np.linalg.LinAlgError:
        if np.any(np.isnan(a)):
            return np.nan
        else:
            return np.inf