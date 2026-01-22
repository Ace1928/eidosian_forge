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
class loglaplace_gen(rv_continuous):
    """A log-Laplace continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `loglaplace` is:

    .. math::

        f(x, c) = \\begin{cases}\\frac{c}{2} x^{ c-1}  &\\text{for } 0 < x < 1\\\\
                               \\frac{c}{2} x^{-c-1}  &\\text{for } x \\ge 1
                  \\end{cases}

    for :math:`c > 0`.

    `loglaplace` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    References
    ----------
    T.J. Kozubowski and K. Podgorski, "A log-Laplace growth rate model",
    The Mathematical Scientist, vol. 28, pp. 49-60, 2003.

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (False, False))]

    def _pdf(self, x, c):
        cd2 = c / 2.0
        c = np.where(x < 1, c, -c)
        return cd2 * x ** (c - 1)

    def _cdf(self, x, c):
        return np.where(x < 1, 0.5 * x ** c, 1 - 0.5 * x ** (-c))

    def _sf(self, x, c):
        return np.where(x < 1, 1 - 0.5 * x ** c, 0.5 * x ** (-c))

    def _ppf(self, q, c):
        return np.where(q < 0.5, (2.0 * q) ** (1.0 / c), (2 * (1.0 - q)) ** (-1.0 / c))

    def _isf(self, q, c):
        return np.where(q > 0.5, (2.0 * (1.0 - q)) ** (1.0 / c), (2 * q) ** (-1.0 / c))

    def _munp(self, n, c):
        return c ** 2 / (c ** 2 - n ** 2)

    def _entropy(self, c):
        return np.log(2.0 / c) + 1.0