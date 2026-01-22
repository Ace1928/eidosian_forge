import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    