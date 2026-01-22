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
class kappa3_gen(rv_continuous):
    """Kappa 3 parameter distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `kappa3` is:

    .. math::

        f(x, a) = a (a + x^a)^{-(a + 1)/a}

    for :math:`x > 0` and :math:`a > 0`.

    `kappa3` takes ``a`` as a shape parameter for :math:`a`.

    References
    ----------
    P.W. Mielke and E.S. Johnson, "Three-Parameter Kappa Distribution Maximum
    Likelihood and Likelihood Ratio Tests", Methods in Weather Research,
    701-707, (September, 1973),
    :doi:`10.1175/1520-0493(1973)101<0701:TKDMLE>2.3.CO;2`

    B. Kumphon, "Maximum Entropy and Maximum Likelihood Estimation for the
    Three-Parameter Kappa Distribution", Open Journal of Statistics, vol 2,
    415-419 (2012), :doi:`10.4236/ojs.2012.24050`

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _pdf(self, x, a):
        return a * (a + x ** a) ** (-1.0 / a - 1)

    def _cdf(self, x, a):
        return x * (a + x ** a) ** (-1.0 / a)

    def _sf(self, x, a):
        x, a = np.broadcast_arrays(x, a)
        sf = super()._sf(x, a)
        cutoff = 0.01
        i = sf < cutoff
        sf2 = -sc.expm1(sc.xlog1py(-1.0 / a[i], a[i] * x[i] ** (-a[i])))
        i2 = sf2 > cutoff
        sf2[i2] = sf[i][i2]
        sf[i] = sf2
        return sf

    def _ppf(self, q, a):
        return (a / (q ** (-a) - 1.0)) ** (1.0 / a)

    def _isf(self, q, a):
        lg = sc.xlog1py(-a, -q)
        denom = sc.expm1(lg)
        return (a / denom) ** (1.0 / a)

    def _stats(self, a):
        outputs = [None if np.any(i < a) else np.nan for i in range(1, 5)]
        return outputs[:]

    def _mom1_sc(self, m, *args):
        if np.any(m >= args[0]):
            return np.nan
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,) + args)[0]