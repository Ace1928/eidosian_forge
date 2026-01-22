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
class dweibull_gen(rv_continuous):
    """A double Weibull continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `dweibull` is given by

    .. math::

        f(x, c) = c / 2 |x|^{c-1} \\exp(-|x|^c)

    for a real number :math:`x` and :math:`c > 0`.

    `dweibull` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _rvs(self, c, size=None, random_state=None):
        u = random_state.uniform(size=size)
        w = weibull_min.rvs(c, size=size, random_state=random_state)
        return w * np.where(u >= 0.5, 1, -1)

    def _pdf(self, x, c):
        ax = abs(x)
        Px = c / 2.0 * ax ** (c - 1.0) * np.exp(-ax ** c)
        return Px

    def _logpdf(self, x, c):
        ax = abs(x)
        return np.log(c) - np.log(2.0) + sc.xlogy(c - 1.0, ax) - ax ** c

    def _cdf(self, x, c):
        Cx1 = 0.5 * np.exp(-abs(x) ** c)
        return np.where(x > 0, 1 - Cx1, Cx1)

    def _ppf(self, q, c):
        fac = 2.0 * np.where(q <= 0.5, q, 1.0 - q)
        fac = np.power(-np.log(fac), 1.0 / c)
        return np.where(q > 0.5, fac, -fac)

    def _sf(self, x, c):
        half_weibull_min_sf = 0.5 * stats.weibull_min._sf(np.abs(x), c)
        return np.where(x > 0, half_weibull_min_sf, 1 - half_weibull_min_sf)

    def _isf(self, q, c):
        double_q = 2.0 * np.where(q <= 0.5, q, 1.0 - q)
        weibull_min_isf = stats.weibull_min._isf(double_q, c)
        return np.where(q > 0.5, -weibull_min_isf, weibull_min_isf)

    def _munp(self, n, c):
        return (1 - n % 2) * sc.gamma(1.0 + 1.0 * n / c)

    def _stats(self, c):
        return (0, None, 0, None)

    def _entropy(self, c):
        h = stats.weibull_min._entropy(c) - np.log(0.5)
        return h