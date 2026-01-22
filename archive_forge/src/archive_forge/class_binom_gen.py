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
class binom_gen(rv_discrete):
    """A binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `binom` is:

    .. math::

       f(k) = \\binom{n}{k} p^k (1-p)^{n-k}

    for :math:`k \\in \\{0, 1, \\dots, n\\}`, :math:`0 \\leq p \\leq 1`

    `binom` takes :math:`n` and :math:`p` as shape parameters,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, nbinom, nhypergeom

    """

    def _shape_info(self):
        return [_ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('p', False, (0, 1), (True, True))]

    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n >= 0) & _isintegral(n) & (p >= 0) & (p <= 1)

    def _get_support(self, n, p):
        return (self.a, n)

    def _logpmf(self, x, n, p):
        k = floor(x)
        combiln = gamln(n + 1) - (gamln(k + 1) + gamln(n - k + 1))
        return combiln + special.xlogy(k, p) + special.xlog1py(n - k, -p)

    def _pmf(self, x, n, p):
        return _boost._binom_pdf(x, n, p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return _boost._binom_cdf(k, n, p)

    def _sf(self, x, n, p):
        k = floor(x)
        return _boost._binom_sf(k, n, p)

    def _isf(self, x, n, p):
        return _boost._binom_isf(x, n, p)

    def _ppf(self, q, n, p):
        return _boost._binom_ppf(q, n, p)

    def _stats(self, n, p, moments='mv'):
        mu = _boost._binom_mean(n, p)
        var = _boost._binom_variance(n, p)
        g1, g2 = (None, None)
        if 's' in moments:
            g1 = _boost._binom_skewness(n, p)
        if 'k' in moments:
            g2 = _boost._binom_kurtosis_excess(n, p)
        return (mu, var, g1, g2)

    def _entropy(self, n, p):
        k = np.r_[0:n + 1]
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)