import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
class nakagami_gen(rv_continuous):
    """A Nakagami continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `nakagami` is:

    .. math::

        f(x, \\nu) = \\frac{2 \\nu^\\nu}{\\Gamma(\\nu)} x^{2\\nu-1} \\exp(-\\nu x^2)

    for :math:`x >= 0`, :math:`\\nu > 0`. The distribution was introduced in
    [2]_, see also [1]_ for further information.

    `nakagami` takes ``nu`` as a shape parameter for :math:`\\nu`.

    %(after_notes)s

    References
    ----------
    .. [1] "Nakagami distribution", Wikipedia
           https://en.wikipedia.org/wiki/Nakagami_distribution
    .. [2] M. Nakagami, "The m-distribution - A general formula of intensity
           distribution of rapid fading", Statistical methods in radio wave
           propagation, Pergamon Press, 1960, 3-36.
           :doi:`10.1016/B978-0-08-009306-2.50005-4`

    %(example)s

    """

    def _argcheck(self, nu):
        return nu > 0

    def _shape_info(self):
        return [_ShapeInfo('nu', False, (0, np.inf), (False, False))]

    def _pdf(self, x, nu):
        return np.exp(self._logpdf(x, nu))

    def _logpdf(self, x, nu):
        return np.log(2) + sc.xlogy(nu, nu) - sc.gammaln(nu) + sc.xlogy(2 * nu - 1, x) - nu * x ** 2

    def _cdf(self, x, nu):
        return sc.gammainc(nu, nu * x * x)

    def _ppf(self, q, nu):
        return np.sqrt(1.0 / nu * sc.gammaincinv(nu, q))

    def _sf(self, x, nu):
        return sc.gammaincc(nu, nu * x * x)

    def _isf(self, p, nu):
        return np.sqrt(1 / nu * sc.gammainccinv(nu, p))

    def _stats(self, nu):
        mu = sc.poch(nu, 0.5) / np.sqrt(nu)
        mu2 = 1.0 - mu * mu
        g1 = mu * (1 - 4 * nu * mu2) / 2.0 / nu / np.power(mu2, 1.5)
        g2 = -6 * mu ** 4 * nu + (8 * nu - 2) * mu ** 2 - 2 * nu + 1
        g2 /= nu * mu2 ** 2.0
        return (mu, mu2, g1, g2)

    def _entropy(self, nu):
        shape = np.shape(nu)
        nu = np.atleast_1d(nu)
        A = sc.gammaln(nu)
        B = nu - (nu - 0.5) * sc.digamma(nu)
        C = -0.5 * np.log(nu) - np.log(2)
        h = A + B + C
        norm_entropy = stats.norm._entropy()
        i = nu > 50000.0
        h[i] = C[i] + norm_entropy - 1 / (12 * nu[i])
        return h.reshape(shape)[()]

    def _rvs(self, nu, size=None, random_state=None):
        return np.sqrt(random_state.standard_gamma(nu, size=size) / nu)

    def _fitstart(self, data, args=None):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        if args is None:
            args = (1.0,) * self.numargs
        loc = np.min(data)
        scale = np.sqrt(np.sum((data - loc) ** 2) / len(data))
        return args + (loc, scale)