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
class planck_gen(rv_discrete):
    """A Planck discrete exponential random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `planck` is:

    .. math::

        f(k) = (1-\\exp(-\\lambda)) \\exp(-\\lambda k)

    for :math:`k \\ge 0` and :math:`\\lambda > 0`.

    `planck` takes :math:`\\lambda` as shape parameter. The Planck distribution
    can be written as a geometric distribution (`geom`) with
    :math:`p = 1 - \\exp(-\\lambda)` shifted by ``loc = -1``.

    %(after_notes)s

    See Also
    --------
    geom

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('lambda', False, (0, np.inf), (False, False))]

    def _argcheck(self, lambda_):
        return lambda_ > 0

    def _pmf(self, k, lambda_):
        return -expm1(-lambda_) * exp(-lambda_ * k)

    def _cdf(self, x, lambda_):
        k = floor(x)
        return -expm1(-lambda_ * (k + 1))

    def _sf(self, x, lambda_):
        return exp(self._logsf(x, lambda_))

    def _logsf(self, x, lambda_):
        k = floor(x)
        return -lambda_ * (k + 1)

    def _ppf(self, q, lambda_):
        vals = ceil(-1.0 / lambda_ * log1p(-q) - 1)
        vals1 = (vals - 1).clip(*self._get_support(lambda_))
        temp = self._cdf(vals1, lambda_)
        return np.where(temp >= q, vals1, vals)

    def _rvs(self, lambda_, size=None, random_state=None):
        p = -expm1(-lambda_)
        return random_state.geometric(p, size=size) - 1.0

    def _stats(self, lambda_):
        mu = 1 / expm1(lambda_)
        var = exp(-lambda_) / expm1(-lambda_) ** 2
        g1 = 2 * cosh(lambda_ / 2.0)
        g2 = 4 + 2 * cosh(lambda_)
        return (mu, var, g1, g2)

    def _entropy(self, lambda_):
        C = -expm1(-lambda_)
        return lambda_ * exp(-lambda_) / C - log(C)