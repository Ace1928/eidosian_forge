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
class anglit_gen(rv_continuous):
    """An anglit continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `anglit` is:

    .. math::

        f(x) = \\sin(2x + \\pi/2) = \\cos(2x)

    for :math:`-\\pi/4 \\le x \\le \\pi/4`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return np.cos(2 * x)

    def _cdf(self, x):
        return np.sin(x + np.pi / 4) ** 2.0

    def _sf(self, x):
        return np.cos(x + np.pi / 4) ** 2.0

    def _ppf(self, q):
        return np.arcsin(np.sqrt(q)) - np.pi / 4

    def _stats(self):
        return (0.0, np.pi * np.pi / 16 - 0.5, 0.0, -2 * (np.pi ** 4 - 96) / (np.pi * np.pi - 8) ** 2)

    def _entropy(self):
        return 1 - np.log(2)