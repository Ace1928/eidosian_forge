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
class fatiguelife_gen(rv_continuous):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `fatiguelife` is:

    .. math::

        f(x, c) = \\frac{x+1}{2c\\sqrt{2\\pi x^3}} \\exp(-\\frac{(x-1)^2}{2x c^2})

    for :math:`x >= 0` and :math:`c > 0`.

    `fatiguelife` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    .. [1] "Birnbaum-Saunders distribution",
           https://en.wikipedia.org/wiki/Birnbaum-Saunders_distribution

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _rvs(self, c, size=None, random_state=None):
        z = random_state.standard_normal(size)
        x = 0.5 * c * z
        x2 = x * x
        t = 1.0 + 2 * x2 + 2 * x * np.sqrt(1 + x2)
        return t

    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        return np.log(x + 1) - (x - 1) ** 2 / (2.0 * x * c ** 2) - np.log(2 * c) - 0.5 * (np.log(2 * np.pi) + 3 * np.log(x))

    def _cdf(self, x, c):
        return _norm_cdf(1.0 / c * (np.sqrt(x) - 1.0 / np.sqrt(x)))

    def _ppf(self, q, c):
        tmp = c * _norm_ppf(q)
        return 0.25 * (tmp + np.sqrt(tmp ** 2 + 4)) ** 2

    def _sf(self, x, c):
        return _norm_sf(1.0 / c * (np.sqrt(x) - 1.0 / np.sqrt(x)))

    def _isf(self, q, c):
        tmp = -c * _norm_ppf(q)
        return 0.25 * (tmp + np.sqrt(tmp ** 2 + 4)) ** 2

    def _stats(self, c):
        c2 = c * c
        mu = c2 / 2.0 + 1.0
        den = 5.0 * c2 + 4.0
        mu2 = c2 * den / 4.0
        g1 = 4 * c * (11 * c2 + 6.0) / np.power(den, 1.5)
        g2 = 6 * c2 * (93 * c2 + 40.0) / den ** 2.0
        return (mu, mu2, g1, g2)