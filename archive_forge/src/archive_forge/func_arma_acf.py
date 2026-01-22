from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def arma_acf(ar, ma, lags=10):
    """
    Theoretical autocorrelation function of an ARMA process.

    Parameters
    ----------
    ar : array_like
        Coefficients for autoregressive lag polynomial, including zero lag.
    ma : array_like
        Coefficients for moving-average lag polynomial, including zero lag.
    lags : int
        The number of terms (lags plus zero lag) to include in returned acf.

    Returns
    -------
    ndarray
        The autocorrelations of ARMA process given by ar and ma.

    See Also
    --------
    arma_acovf : Autocovariances from ARMA processes.
    acf : Sample autocorrelation function estimation.
    acovf : Sample autocovariance function estimation.
    """
    acovf = arma_acovf(ar, ma, lags)
    return acovf / acovf[0]