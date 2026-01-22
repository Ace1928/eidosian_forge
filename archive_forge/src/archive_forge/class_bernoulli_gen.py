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
class bernoulli_gen(binom_gen):
    """A Bernoulli discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \\begin{cases}1-p  &\\text{if } k = 0\\\\
                           p    &\\text{if } k = 1\\end{cases}

    for :math:`k` in :math:`\\{0, 1\\}`, :math:`0 \\leq p \\leq 1`

    `bernoulli` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('p', False, (0, 1), (True, True))]

    def _rvs(self, p, size=None, random_state=None):
        return binom_gen._rvs(self, 1, p, size=size, random_state=random_state)

    def _argcheck(self, p):
        return (p >= 0) & (p <= 1)

    def _get_support(self, p):
        return (self.a, self.b)

    def _logpmf(self, x, p):
        return binom._logpmf(x, 1, p)

    def _pmf(self, x, p):
        return binom._pmf(x, 1, p)

    def _cdf(self, x, p):
        return binom._cdf(x, 1, p)

    def _sf(self, x, p):
        return binom._sf(x, 1, p)

    def _isf(self, x, p):
        return binom._isf(x, 1, p)

    def _ppf(self, q, p):
        return binom._ppf(q, 1, p)

    def _stats(self, p):
        return binom._stats(1, p)

    def _entropy(self, p):
        return entr(p) + entr(1 - p)