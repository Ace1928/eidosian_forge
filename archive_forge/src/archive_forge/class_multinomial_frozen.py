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
class multinomial_frozen(multi_rv_frozen):
    """Create a frozen Multinomial distribution.

    Parameters
    ----------
    n : int
        number of trials
    p: array_like
        probability of a trial falling into each category; should sum to 1
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    """

    def __init__(self, n, p, seed=None):
        self._dist = multinomial_gen(seed)
        self.n, self.p, self.npcond = self._dist._process_parameters(n, p)

        def _process_parameters(n, p):
            return (self.n, self.p, self.npcond)
        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, self.n, self.p)

    def pmf(self, x):
        return self._dist.pmf(x, self.n, self.p)

    def mean(self):
        return self._dist.mean(self.n, self.p)

    def cov(self):
        return self._dist.cov(self.n, self.p)

    def entropy(self):
        return self._dist.entropy(self.n, self.p)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.n, self.p, size, random_state)