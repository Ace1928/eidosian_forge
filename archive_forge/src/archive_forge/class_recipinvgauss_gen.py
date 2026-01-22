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
class recipinvgauss_gen(rv_continuous):
    """A reciprocal inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `recipinvgauss` is:

    .. math::

        f(x, \\mu) = \\frac{1}{\\sqrt{2\\pi x}}
                    \\exp\\left(\\frac{-(1-\\mu x)^2}{2\\mu^2x}\\right)

    for :math:`x \\ge 0`.

    `recipinvgauss` takes ``mu`` as a shape parameter for :math:`\\mu`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('mu', False, (0, np.inf), (False, False))]

    def _pdf(self, x, mu):
        return np.exp(self._logpdf(x, mu))

    def _logpdf(self, x, mu):
        return _lazywhere(x > 0, (x, mu), lambda x, mu: -(1 - mu * x) ** 2.0 / (2 * x * mu ** 2.0) - 0.5 * np.log(2 * np.pi * x), fillvalue=-np.inf)

    def _cdf(self, x, mu):
        trm1 = 1.0 / mu - x
        trm2 = 1.0 / mu + x
        isqx = 1.0 / np.sqrt(x)
        return _norm_cdf(-isqx * trm1) - np.exp(2.0 / mu) * _norm_cdf(-isqx * trm2)

    def _sf(self, x, mu):
        trm1 = 1.0 / mu - x
        trm2 = 1.0 / mu + x
        isqx = 1.0 / np.sqrt(x)
        return _norm_cdf(isqx * trm1) + np.exp(2.0 / mu) * _norm_cdf(-isqx * trm2)

    def _rvs(self, mu, size=None, random_state=None):
        return 1.0 / random_state.wald(mu, 1.0, size=size)