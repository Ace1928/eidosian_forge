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
class random_table_frozen(multi_rv_frozen):

    def __init__(self, row, col, *, seed=None):
        self._dist = random_table_gen(seed)
        self._params = self._dist._process_parameters(row, col)

        def _process_parameters(r, c):
            return self._params
        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, None, None)

    def pmf(self, x):
        return self._dist.pmf(x, None, None)

    def mean(self):
        return self._dist.mean(None, None)

    def rvs(self, size=None, method=None, random_state=None):
        return self._dist.rvs(None, None, size=size, method=method, random_state=random_state)