import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_multivariate_pacf_from_coefficients(constrained, error_variance, order=None, k_endog=None):
    """
    Transform matrices corresponding to a stationary (or invertible) process
    to matrices with singular values less than one.

    Parameters
    ----------
    constrained : array or list
        The coefficients matrices. If a list, should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`. If
        an array, should be the coefficient matrices horizontally concatenated
        and sized `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    pacf : list
        List of first `order` multivariate partial autocorrelations.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Note that this computes multivariate partial autocorrelations.

    Corresponds to the inverse of Lemma 2.1 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.

    Notes
    -----
    Coefficients are assumed to be provided from the VAR model:

    .. math::
        y_t = A_1 y_{t-1} + \\dots + A_p y_{t-p} + \\varepsilon_t
    """
    if type(constrained) is list:
        order = len(constrained)
        k_endog = constrained[0].shape[0]
    else:
        k_endog, order = constrained.shape
        order //= k_endog
    _acovf = _compute_multivariate_acovf_from_coefficients
    autocovariances = [autocovariance.T for autocovariance in _acovf(constrained, error_variance, maxlag=order)]
    return _compute_multivariate_pacf_from_autocovariances(autocovariances)