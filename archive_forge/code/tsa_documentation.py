import numpy as np
autocovariance function with call to fftconvolve, biased

    Parameters
    ----------
    x : array_like
        timeseries, signal
    demean : bool
        If true, then demean time series

    Returns
    -------
    acovf : ndarray
        autocovariance for data, same length as x

    might work for nd in parallel with time along axis 0

    