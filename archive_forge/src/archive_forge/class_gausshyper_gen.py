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
class gausshyper_gen(rv_continuous):
    """A Gauss hypergeometric continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gausshyper` is:

    .. math::

        f(x, a, b, c, z) = C x^{a-1} (1-x)^{b-1} (1+zx)^{-c}

    for :math:`0 \\le x \\le 1`, :math:`a,b > 0`, :math:`c` a real number,
    :math:`z > -1`, and :math:`C = \\frac{1}{B(a, b) F[2, 1](c, a; a+b; -z)}`.
    :math:`F[2, 1]` is the Gauss hypergeometric function
    `scipy.special.hyp2f1`.

    `gausshyper` takes :math:`a`, :math:`b`, :math:`c` and :math:`z` as shape
    parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Armero, C., and M. J. Bayarri. "Prior Assessments for Prediction in
           Queues." *Journal of the Royal Statistical Society*. Series D (The
           Statistician) 43, no. 1 (1994): 139-53. doi:10.2307/2348939

    %(example)s

    """

    def _argcheck(self, a, b, c, z):
        return (a > 0) & (b > 0) & (c == c) & (z > -1)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        ic = _ShapeInfo('c', False, (-np.inf, np.inf), (False, False))
        iz = _ShapeInfo('z', False, (-1, np.inf), (False, False))
        return [ia, ib, ic, iz]

    def _pdf(self, x, a, b, c, z):
        normalization_constant = sc.beta(a, b) * sc.hyp2f1(c, a, a + b, -z)
        return 1.0 / normalization_constant * x ** (a - 1.0) * (1.0 - x) ** (b - 1.0) / (1.0 + z * x) ** c

    def _munp(self, n, a, b, c, z):
        fac = sc.beta(n + a, b) / sc.beta(a, b)
        num = sc.hyp2f1(c, a + n, a + b + n, -z)
        den = sc.hyp2f1(c, a, a + b, -z)
        return fac * num / den