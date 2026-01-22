import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _get_base_step(dtype=np.float64):
    fmin = 4 * np.finfo(dtype).tiny
    tmax = np.arcsinh(np.log(2 / fmin - 1) / np.pi)
    h0 = tmax / _N_BASE_STEPS
    return h0.astype(dtype)