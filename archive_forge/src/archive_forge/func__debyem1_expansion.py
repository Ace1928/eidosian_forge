import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def _debyem1_expansion(x):
    """Debye function minus 1, Taylor series approximation around zero

    function is not used
    """
    x = np.asarray(x)
    dm1 = -x / 4 + x ** 2 / 36 - x ** 4 / 3600 + x ** 6 / 211680 - x ** 8 / 10886400 + x ** 10 / 526901760 - x ** 12 * 691 / 16999766784000
    return dm1