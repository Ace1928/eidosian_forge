import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def argus_gamma_trf(x, chi):
    if chi <= 5:
        return x
    return np.sqrt(1.0 - 2 * x / chi ** 2)