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
class gengamma_gen(rv_continuous):
    """A generalized gamma continuous random variable.

    %(before_notes)s

    See Also
    --------
    gamma, invgamma, weibull_min

    Notes
    -----
    The probability density function for `gengamma` is ([1]_):

    .. math::

        f(x, a, c) = \\frac{|c| x^{c a-1} \\exp(-x^c)}{\\Gamma(a)}

    for :math:`x \\ge 0`, :math:`a > 0`, and :math:`c \\ne 0`.
    :math:`\\Gamma` is the gamma function (`scipy.special.gamma`).

    `gengamma` takes :math:`a` and :math:`c` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] E.W. Stacy, "A Generalization of the Gamma Distribution",
       Annals of Mathematical Statistics, Vol 33(3), pp. 1187--1192.

    %(example)s

    """

    def _argcheck(self, a, c):
        return (a > 0) & (c != 0)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ic = _ShapeInfo('c', False, (-np.inf, np.inf), (False, False))
        return [ia, ic]

    def _pdf(self, x, a, c):
        return np.exp(self._logpdf(x, a, c))

    def _logpdf(self, x, a, c):
        return _lazywhere((x != 0) | (c > 0), (x, c), lambda x, c: np.log(abs(c)) + sc.xlogy(c * a - 1, x) - x ** c - sc.gammaln(a), fillvalue=-np.inf)

    def _cdf(self, x, a, c):
        xc = x ** c
        val1 = sc.gammainc(a, xc)
        val2 = sc.gammaincc(a, xc)
        return np.where(c > 0, val1, val2)

    def _rvs(self, a, c, size=None, random_state=None):
        r = random_state.standard_gamma(a, size=size)
        return r ** (1.0 / c)

    def _sf(self, x, a, c):
        xc = x ** c
        val1 = sc.gammainc(a, xc)
        val2 = sc.gammaincc(a, xc)
        return np.where(c > 0, val2, val1)

    def _ppf(self, q, a, c):
        val1 = sc.gammaincinv(a, q)
        val2 = sc.gammainccinv(a, q)
        return np.where(c > 0, val1, val2) ** (1.0 / c)

    def _isf(self, q, a, c):
        val1 = sc.gammaincinv(a, q)
        val2 = sc.gammainccinv(a, q)
        return np.where(c > 0, val2, val1) ** (1.0 / c)

    def _munp(self, n, a, c):
        return sc.poch(a, n * 1.0 / c)

    def _entropy(self, a, c):

        def regular(a, c):
            val = sc.psi(a)
            A = a * (1 - val) + val / c
            B = sc.gammaln(a) - np.log(abs(c))
            h = A + B
            return h

        def asymptotic(a, c):
            return norm._entropy() - np.log(a) / 2 - np.log(np.abs(c)) + a ** (-1.0) / 6 - a ** (-3.0) / 90 + (np.log(a) - a ** (-1.0) / 2 - a ** (-2.0) / 12 + a ** (-4.0) / 120) / c
        h = _lazywhere(a >= 200.0, (a, c), f=asymptotic, f2=regular)
        return h