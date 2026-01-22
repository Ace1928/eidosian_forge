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
class truncexpon_gen(rv_continuous):
    """A truncated exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `truncexpon` is:

    .. math::

        f(x, b) = \\frac{\\exp(-x)}{1 - \\exp(-b)}

    for :math:`0 <= x <= b`.

    `truncexpon` takes ``b`` as a shape parameter for :math:`b`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('b', False, (0, np.inf), (False, False))]

    def _get_support(self, b):
        return (self.a, b)

    def _pdf(self, x, b):
        return np.exp(-x) / -sc.expm1(-b)

    def _logpdf(self, x, b):
        return -x - np.log(-sc.expm1(-b))

    def _cdf(self, x, b):
        return sc.expm1(-x) / sc.expm1(-b)

    def _ppf(self, q, b):
        return -sc.log1p(q * sc.expm1(-b))

    def _sf(self, x, b):
        return (np.exp(-b) - np.exp(-x)) / sc.expm1(-b)

    def _isf(self, q, b):
        return -np.log(np.exp(-b) - q * sc.expm1(-b))

    def _munp(self, n, b):
        if n == 1:
            return (1 - (b + 1) * np.exp(-b)) / -sc.expm1(-b)
        elif n == 2:
            return 2 * (1 - 0.5 * (b * b + 2 * b + 2) * np.exp(-b)) / -sc.expm1(-b)
        else:
            return super()._munp(n, b)

    def _entropy(self, b):
        eB = np.exp(b)
        return np.log(eB - 1) + (1 + eB * (b - 1.0)) / (1.0 - eB)