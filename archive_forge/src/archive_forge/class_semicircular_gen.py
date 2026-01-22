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
class semicircular_gen(rv_continuous):
    """A semicircular continuous random variable.

    %(before_notes)s

    See Also
    --------
    rdist

    Notes
    -----
    The probability density function for `semicircular` is:

    .. math::

        f(x) = \\frac{2}{\\pi} \\sqrt{1-x^2}

    for :math:`-1 \\le x \\le 1`.

    The distribution is a special case of `rdist` with `c = 3`.

    %(after_notes)s

    References
    ----------
    .. [1] "Wigner semicircle distribution",
           https://en.wikipedia.org/wiki/Wigner_semicircle_distribution

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return 2.0 / np.pi * np.sqrt(1 - x * x)

    def _logpdf(self, x):
        return np.log(2 / np.pi) + 0.5 * sc.log1p(-x * x)

    def _cdf(self, x):
        return 0.5 + 1.0 / np.pi * (x * np.sqrt(1 - x * x) + np.arcsin(x))

    def _ppf(self, q):
        return rdist._ppf(q, 3)

    def _rvs(self, size=None, random_state=None):
        r = np.sqrt(random_state.uniform(size=size))
        a = np.cos(np.pi * random_state.uniform(size=size))
        return r * a

    def _stats(self):
        return (0, 0.25, 0, -1.0)

    def _entropy(self):
        return 0.6447298858494002