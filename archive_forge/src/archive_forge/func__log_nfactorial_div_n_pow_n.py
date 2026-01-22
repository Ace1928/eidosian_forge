import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _log_nfactorial_div_n_pow_n(n):
    rn = 1.0 / n
    return np.log(n) / 2 - n + _LOG_2PI / 2 + rn * np.polyval(_STIRLING_COEFFS, rn / n)