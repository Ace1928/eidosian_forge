import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def argus_pdf(x, chi):
    if chi <= 5:
        y = 1 - x * x
        return x * math.sqrt(y) * math.exp(-0.5 * chi ** 2 * y)
    return math.sqrt(x) * math.exp(-x)