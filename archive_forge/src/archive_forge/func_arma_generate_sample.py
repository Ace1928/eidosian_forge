from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def arma_generate_sample(ar, ma, nsample, scale=1, distrvs=None, axis=0, burnin=0):
    """
    Simulate data from an ARMA.

    Parameters
    ----------
    ar : array_like
        The coefficient for autoregressive lag polynomial, including zero lag.
    ma : array_like
        The coefficient for moving-average lag polynomial, including zero lag.
    nsample : int or tuple of ints
        If nsample is an integer, then this creates a 1d timeseries of
        length size. If nsample is a tuple, creates a len(nsample)
        dimensional time series where time is indexed along the input
        variable ``axis``. All series are unless ``distrvs`` generates
        dependent data.
    scale : float
        The standard deviation of noise.
    distrvs : function, random number generator
        A function that generates the random numbers, and takes ``size``
        as argument. The default is np.random.standard_normal.
    axis : int
        See nsample for details.
    burnin : int
        Number of observation at the beginning of the sample to drop.
        Used to reduce dependence on initial values.

    Returns
    -------
    ndarray
        Random sample(s) from an ARMA process.

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR parameters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARIMA(y, (2, 0, 2), trend='n').fit(disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    distrvs = np.random.standard_normal if distrvs is None else distrvs
    if np.ndim(nsample) == 0:
        nsample = [nsample]
    if burnin:
        newsize = list(nsample)
        newsize[axis] += burnin
        newsize = tuple(newsize)
        fslice = [slice(None)] * len(newsize)
        fslice[axis] = slice(burnin, None, None)
        fslice = tuple(fslice)
    else:
        newsize = tuple(nsample)
        fslice = tuple([slice(None)] * np.ndim(newsize))
    eta = scale * distrvs(size=newsize)
    return signal.lfilter(ma, ar, eta, axis=axis)[fslice]