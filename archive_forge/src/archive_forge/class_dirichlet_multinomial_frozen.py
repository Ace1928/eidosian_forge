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
class dirichlet_multinomial_frozen(multi_rv_frozen):

    def __init__(self, alpha, n, seed=None):
        alpha, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        self.alpha = alpha
        self.n = n
        self._dist = dirichlet_multinomial_gen(seed)

    def logpmf(self, x):
        return self._dist.logpmf(x, self.alpha, self.n)

    def pmf(self, x):
        return self._dist.pmf(x, self.alpha, self.n)

    def mean(self):
        return self._dist.mean(self.alpha, self.n)

    def var(self):
        return self._dist.var(self.alpha, self.n)

    def cov(self):
        return self._dist.cov(self.alpha, self.n)