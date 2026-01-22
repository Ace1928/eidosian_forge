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
class invweibull_gen(rv_continuous):
    """An inverted Weibull continuous random variable.

    This distribution is also known as the Fréchet distribution or the
    type II extreme value distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invweibull` is:

    .. math::

        f(x, c) = c x^{-c-1} \\exp(-x^{-c})

    for :math:`x > 0`, :math:`c > 0`.

    `invweibull` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    F.R.S. de Gusmao, E.M.M Ortega and G.M. Cordeiro, "The generalized inverse
    Weibull distribution", Stat. Papers, vol. 52, pp. 591-619, 2011.

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        xc1 = np.power(x, -c - 1.0)
        xc2 = np.power(x, -c)
        xc2 = np.exp(-xc2)
        return c * xc1 * xc2

    def _cdf(self, x, c):
        xc1 = np.power(x, -c)
        return np.exp(-xc1)

    def _sf(self, x, c):
        return -np.expm1(-x ** (-c))

    def _ppf(self, q, c):
        return np.power(-np.log(q), -1.0 / c)

    def _isf(self, p, c):
        return (-np.log1p(-p)) ** (-1 / c)

    def _munp(self, n, c):
        return sc.gamma(1 - n / c)

    def _entropy(self, c):
        return 1 + _EULER + _EULER / c - np.log(c)

    def _fitstart(self, data, args=None):
        args = (2.0,) if args is None else args
        return super()._fitstart(data, args=args)