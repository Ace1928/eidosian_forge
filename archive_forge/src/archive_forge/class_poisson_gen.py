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
class poisson_gen(rv_discrete):
    """A Poisson discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \\exp(-\\mu) \\frac{\\mu^k}{k!}

    for :math:`k \\ge 0`.

    `poisson` takes :math:`\\mu \\geq 0` as shape parameter.
    When :math:`\\mu = 0`, the ``pmf`` method
    returns ``1.0`` at quantile :math:`k = 0`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('mu', False, (0, np.inf), (True, False))]

    def _argcheck(self, mu):
        return mu >= 0

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    def _logpmf(self, k, mu):
        Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
        return Pk

    def _pmf(self, k, mu):
        return exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = floor(x)
        return special.pdtr(k, mu)

    def _sf(self, x, mu):
        k = floor(x)
        return special.pdtrc(k, mu)

    def _ppf(self, q, mu):
        vals = ceil(special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: sqrt(1.0 / x), np.inf)
        g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0 / x, np.inf)
        return (mu, var, g1, g2)