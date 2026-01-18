import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def unconstrain_stationary_multivariate(constrained, error_variance):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : array or list
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer. If a list, should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`. If an array, should be
        the coefficient matrices horizontally concatenated and sized
        `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False).

    Returns
    -------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component. Will match the type of the passed `constrained`
        variable (so if a list was passed, a list will be returned).

    Notes
    -----
    Uses the list representation internally, even if an array is passed.

    References
    ----------
    .. [*] Ansley, Craig F., and Robert Kohn. 1986.
       "A Note on Reparameterizing a Vector Autoregressive Moving Average Model
       to Enforce Stationarity."
       Journal of Statistical Computation and Simulation 24 (2): 99-106.
    """
    use_list = type(constrained) is list
    if not use_list:
        k_endog, order = constrained.shape
        order //= k_endog
        constrained = [constrained[:k_endog, i * k_endog:(i + 1) * k_endog] for i in range(order)]
    else:
        order = len(constrained)
        k_endog = constrained[0].shape[0]
    partial_autocorrelations = _compute_multivariate_pacf_from_coefficients(constrained, error_variance, order, k_endog)
    unconstrained = _unconstrain_sv_less_than_one(partial_autocorrelations, order, k_endog)
    if not use_list:
        unconstrained = np.concatenate(unconstrained, axis=1)
    return (unconstrained, error_variance)