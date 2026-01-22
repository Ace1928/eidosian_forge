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
class hypsecant_gen(rv_continuous):
    """A hyperbolic secant continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `hypsecant` is:

    .. math::

        f(x) = \\frac{1}{\\pi} \\text{sech}(x)

    for a real number :math:`x`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return 1.0 / (np.pi * np.cosh(x))

    def _cdf(self, x):
        return 2.0 / np.pi * np.arctan(np.exp(x))

    def _ppf(self, q):
        return np.log(np.tan(np.pi * q / 2.0))

    def _sf(self, x):
        return 2.0 / np.pi * np.arctan(np.exp(-x))

    def _isf(self, q):
        return -np.log(np.tan(np.pi * q / 2.0))

    def _stats(self):
        return (0, np.pi * np.pi / 4, 0, 2)

    def _entropy(self):
        return np.log(2 * np.pi)