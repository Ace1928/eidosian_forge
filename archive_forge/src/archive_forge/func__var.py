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
def _var(self, dim, df, scale):
    """Variance of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.

        """
    if df > dim + 3:
        var = (df - dim + 1) * scale ** 2
        diag = scale.diagonal()
        var += (df - dim - 1) * np.outer(diag, diag)
        var /= (df - dim) * (df - dim - 1) ** 2 * (df - dim - 3)
    else:
        var = None
    return var