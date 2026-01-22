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
class wrapcauchy_gen(rv_continuous):
    """A wrapped Cauchy continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `wrapcauchy` is:

    .. math::

        f(x, c) = \\frac{1-c^2}{2\\pi (1+c^2 - 2c \\cos(x))}

    for :math:`0 \\le x \\le 2\\pi`, :math:`0 < c < 1`.

    `wrapcauchy` takes ``c`` as a shape parameter for :math:`c`.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, c):
        return (c > 0) & (c < 1)

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, 1), (False, False))]

    def _pdf(self, x, c):
        return (1.0 - c * c) / (2 * np.pi * (1 + c * c - 2 * c * np.cos(x)))

    def _cdf(self, x, c):

        def f1(x, cr):
            return 1 / np.pi * np.arctan(cr * np.tan(x / 2))

        def f2(x, cr):
            return 1 - 1 / np.pi * np.arctan(cr * np.tan((2 * np.pi - x) / 2))
        cr = (1 + c) / (1 - c)
        return _lazywhere(x < np.pi, (x, cr), f=f1, f2=f2)

    def _ppf(self, q, c):
        val = (1.0 - c) / (1.0 + c)
        rcq = 2 * np.arctan(val * np.tan(np.pi * q))
        rcmq = 2 * np.pi - 2 * np.arctan(val * np.tan(np.pi * (1 - q)))
        return np.where(q < 1.0 / 2, rcq, rcmq)

    def _entropy(self, c):
        return np.log(2 * np.pi * (1 - c * c))

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        return (0.5, np.min(data), np.ptp(data) / (2 * np.pi))