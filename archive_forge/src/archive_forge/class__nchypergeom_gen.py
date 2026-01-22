from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
class _nchypergeom_gen(rv_discrete):
    """A noncentral hypergeometric discrete random variable.

    For subclassing by nchypergeom_fisher_gen and nchypergeom_wallenius_gen.

    """
    rvs_name = None
    dist = None

    def _shape_info(self):
        return [_ShapeInfo('M', True, (0, np.inf), (True, False)), _ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('N', True, (0, np.inf), (True, False)), _ShapeInfo('odds', False, (0, np.inf), (False, False))]

    def _get_support(self, M, n, N, odds):
        N, m1, n = (M, n, N)
        m2 = N - m1
        x_min = np.maximum(0, n - m2)
        x_max = np.minimum(n, m1)
        return (x_min, x_max)

    def _argcheck(self, M, n, N, odds):
        M, n = (np.asarray(M), np.asarray(n))
        N, odds = (np.asarray(N), np.asarray(odds))
        cond1 = (M.astype(int) == M) & (M >= 0)
        cond2 = (n.astype(int) == n) & (n >= 0)
        cond3 = (N.astype(int) == N) & (N >= 0)
        cond4 = odds > 0
        cond5 = N <= M
        cond6 = n <= M
        return cond1 & cond2 & cond3 & cond4 & cond5 & cond6

    def _rvs(self, M, n, N, odds, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, N, odds, size, random_state):
            length = np.prod(size)
            urn = _PyStochasticLib3()
            rv_gen = getattr(urn, self.rvs_name)
            rvs = rv_gen(N, n, M, odds, length, random_state)
            rvs = rvs.reshape(size)
            return rvs
        return _rvs1(M, n, N, odds, size=size, random_state=random_state)

    def _pmf(self, x, M, n, N, odds):
        x, M, n, N, odds = np.broadcast_arrays(x, M, n, N, odds)
        if x.size == 0:
            return np.empty_like(x)

        @np.vectorize
        def _pmf1(x, M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.probability(x)
        return _pmf1(x, M, n, N, odds)

    def _stats(self, M, n, N, odds, moments):

        @np.vectorize
        def _moments1(M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.moments()
        m, v = _moments1(M, n, N, odds) if 'm' in moments or 'v' in moments else (None, None)
        s, k = (None, None)
        return (m, v, s, k)