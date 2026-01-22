import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def _theta_from_tau(alpha):
    return self.tau(theta=alpha) - tau