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
class foldcauchy_gen(rv_continuous):
    """A folded Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `foldcauchy` is:

    .. math::

        f(x, c) = \\frac{1}{\\pi (1+(x-c)^2)} + \\frac{1}{\\pi (1+(x+c)^2)}

    for :math:`x \\ge 0` and :math:`c \\ge 0`.

    `foldcauchy` takes ``c`` as a shape parameter for :math:`c`.

    %(example)s

    """

    def _argcheck(self, c):
        return c >= 0

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, np.inf), (True, False))]

    def _rvs(self, c, size=None, random_state=None):
        return abs(cauchy.rvs(loc=c, size=size, random_state=random_state))

    def _pdf(self, x, c):
        return 1.0 / np.pi * (1.0 / (1 + (x - c) ** 2) + 1.0 / (1 + (x + c) ** 2))

    def _cdf(self, x, c):
        return 1.0 / np.pi * (np.arctan(x - c) + np.arctan(x + c))

    def _sf(self, x, c):
        return (np.arctan2(1, x - c) + np.arctan2(1, x + c)) / np.pi

    def _stats(self, c):
        return (np.inf, np.inf, np.nan, np.nan)