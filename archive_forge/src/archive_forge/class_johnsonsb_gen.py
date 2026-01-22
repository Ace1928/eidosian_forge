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
class johnsonsb_gen(rv_continuous):
    """A Johnson SB continuous random variable.

    %(before_notes)s

    See Also
    --------
    johnsonsu

    Notes
    -----
    The probability density function for `johnsonsb` is:

    .. math::

        f(x, a, b) = \\frac{b}{x(1-x)}  \\phi(a + b \\log \\frac{x}{1-x} )

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`
    and :math:`x \\in [0,1]`.  :math:`\\phi` is the pdf of the normal
    distribution.

    `johnsonsb` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _argcheck(self, a, b):
        return (b > 0) & (a == a)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(self, x, a, b):
        trm = _norm_pdf(a + b * sc.logit(x))
        return b * 1.0 / (x * (1 - x)) * trm

    def _cdf(self, x, a, b):
        return _norm_cdf(a + b * sc.logit(x))

    def _ppf(self, q, a, b):
        return sc.expit(1.0 / b * (_norm_ppf(q) - a))

    def _sf(self, x, a, b):
        return _norm_sf(a + b * sc.logit(x))

    def _isf(self, q, a, b):
        return sc.expit(1.0 / b * (_norm_isf(q) - a))