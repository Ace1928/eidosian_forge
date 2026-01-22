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
class arcsine_gen(rv_continuous):
    """An arcsine continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `arcsine` is:

    .. math::

        f(x) = \\frac{1}{\\pi \\sqrt{x (1-x)}}

    for :math:`0 < x < 1`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        with np.errstate(divide='ignore'):
            return 1.0 / np.pi / np.sqrt(x * (1 - x))

    def _cdf(self, x):
        return 2.0 / np.pi * np.arcsin(np.sqrt(x))

    def _ppf(self, q):
        return np.sin(np.pi / 2.0 * q) ** 2.0

    def _stats(self):
        mu = 0.5
        mu2 = 1.0 / 8
        g1 = 0
        g2 = -3.0 / 2.0
        return (mu, mu2, g1, g2)

    def _entropy(self):
        return -0.24156447527049044