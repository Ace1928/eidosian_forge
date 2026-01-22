from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def arma_impulse_response(ar, ma, leads=100):
    """
    Compute the impulse response function (MA representation) for ARMA process.

    Parameters
    ----------
    ar : array_like, 1d
        The auto regressive lag polynomial.
    ma : array_like, 1d
        The moving average lag polynomial.
    leads : int
        The number of observations to calculate.

    Returns
    -------
    ndarray
        The impulse response function with nobs elements.

    Notes
    -----
    This is the same as finding the MA representation of an ARMA(p,q).
    By reversing the role of ar and ma in the function arguments, the
    returned result is the AR representation of an ARMA(p,q), i.e

    ma_representation = arma_impulse_response(ar, ma, leads=100)
    ar_representation = arma_impulse_response(ma, ar, leads=100)

    Fully tested against matlab

    Examples
    --------
    AR(1)

    >>> arma_impulse_response([1.0, -0.8], [1.], leads=10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    this is the same as

    >>> 0.8**np.arange(10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    MA(2)

    >>> arma_impulse_response([1.0], [1., 0.5, 0.2], leads=10)
    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

    ARMA(1,2)

    >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], leads=10)
    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])
    """
    impulse = np.zeros(leads)
    impulse[0] = 1.0
    return signal.lfilter(ma, ar, impulse)