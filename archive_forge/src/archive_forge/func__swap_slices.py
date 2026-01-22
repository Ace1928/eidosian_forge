import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _swap_slices(x, slc1, slc2):
    t = x[slc1].copy()
    x[slc1] = x[slc2].copy()
    x[slc2] = t