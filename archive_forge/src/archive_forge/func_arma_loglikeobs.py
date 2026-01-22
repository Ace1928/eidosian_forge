import numpy as np
from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from . import _arma_innovations
def arma_loglikeobs(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute the log-likelihood for each observation assuming an ARMA process.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    ndarray
        Array of loglikelihood values for each observation.
    """
    endog = np.array(endog)
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)
    if prefix is None:
        prefix, dtype, _ = find_best_blas_type([endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]
    endog = np.ascontiguousarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = dtype(sigma2).item()
    func = getattr(_arma_innovations, prefix + 'arma_loglikeobs_fast')
    return func(endog, ar_params, ma_params, sigma2)