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
def _givens_to_1(self, aii, ajj, aij):
    """Computes a 2x2 Givens matrix to put 1's on the diagonal.

        The input matrix is a 2x2 symmetric matrix M = [ aii aij ; aij ajj ].

        The output matrix g is a 2x2 anti-symmetric matrix of the form
        [ c s ; -s c ];  the elements c and s are returned.

        Applying the output matrix to the input matrix (as b=g.T M g)
        results in a matrix with bii=1, provided tr(M) - det(M) >= 1
        and floating point issues do not occur. Otherwise, some other
        valid rotation is returned. When tr(M)==2, also bjj=1.

        """
    aiid = aii - 1.0
    ajjd = ajj - 1.0
    if ajjd == 0:
        return (0.0, 1.0)
    dd = math.sqrt(max(aij ** 2 - aiid * ajjd, 0))
    t = (aij + math.copysign(dd, aij)) / ajjd
    c = 1.0 / math.sqrt(1.0 + t * t)
    if c == 0:
        s = 1.0
    else:
        s = c * t
    return (c, s)