import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def _spdroots(self, arroots, maroots, w):
    """spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)

        Parameters
        ----------
        arroots : ndarray
            roots of ar (denominator) lag-polynomial
        maroots : ndarray
            roots of ma (numerator) lag-polynomial
        w : array_like
            frequencies for which spd is calculated

        Notes
        -----
        this should go into a function
        """
    w = np.atleast_2d(w).T
    cosw = np.cos(w)
    maroots = 1.0 / maroots
    arroots = 1.0 / arroots
    num = 1 + maroots ** 2 - 2 * maroots * cosw
    den = 1 + arroots ** 2 - 2 * arroots * cosw
    hw = 0.5 / np.pi * num.prod(-1) / den.prod(-1)
    return (np.squeeze(hw), w.squeeze())