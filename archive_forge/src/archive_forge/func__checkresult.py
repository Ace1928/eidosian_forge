import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _checkresult(self, result, cond, bad_value):
    result = np.asarray(result)
    if cond.ndim != 0:
        result[cond] = bad_value
    elif cond:
        return bad_value
    if result.ndim == 0:
        return result[()]
    return result