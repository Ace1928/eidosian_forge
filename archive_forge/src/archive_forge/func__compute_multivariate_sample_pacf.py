import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_multivariate_sample_pacf(endog, maxlag):
    """
    Computer multivariate sample partial autocorrelations

    Parameters
    ----------
    endog : array_like
        Sample data on which to compute sample autocovariances. Shaped
        `nobs` x `k_endog`.
    maxlag : int
        Maximum lag for which to calculate sample partial autocorrelations.

    Returns
    -------
    sample_pacf : list
        A list of the first `maxlag` sample partial autocorrelation matrices.
        Each matrix is shaped `k_endog` x `k_endog`.
    """
    sample_autocovariances = _compute_multivariate_sample_acovf(endog, maxlag)
    return _compute_multivariate_pacf_from_autocovariances(sample_autocovariances)