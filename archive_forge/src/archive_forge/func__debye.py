import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def _debye(alpha):
    EPSILON = np.finfo(np.float64).eps * 100

    def integrand(t):
        return np.squeeze(t / (np.exp(t) - 1))
    _alpha = np.squeeze(alpha)
    debye_value = integrate.quad(integrand, EPSILON, _alpha)[0] / _alpha
    return debye_value