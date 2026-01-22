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
class moyal_gen(rv_continuous):
    """A Moyal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `moyal` is:

    .. math::

        f(x) = \\exp(-(x + \\exp(-x))/2) / \\sqrt{2\\pi}

    for a real number :math:`x`.

    %(after_notes)s

    This distribution has utility in high-energy physics and radiation
    detection. It describes the energy loss of a charged relativistic
    particle due to ionization of the medium [1]_. It also provides an
    approximation for the Landau distribution. For an in depth description
    see [2]_. For additional description, see [3]_.

    References
    ----------
    .. [1] J.E. Moyal, "XXX. Theory of ionization fluctuations",
           The London, Edinburgh, and Dublin Philosophical Magazine
           and Journal of Science, vol 46, 263-280, (1955).
           :doi:`10.1080/14786440308521076` (gated)
    .. [2] G. Cordeiro et al., "The beta Moyal: a useful skew distribution",
           International Journal of Research and Reviews in Applied Sciences,
           vol 10, 171-192, (2012).
           http://www.arpapress.com/Volumes/Vol10Issue2/IJRRAS_10_2_02.pdf
    .. [3] C. Walck, "Handbook on Statistical Distributions for
           Experimentalists; International Report SUF-PFY/96-01", Chapter 26,
           University of Stockholm: Stockholm, Sweden, (2007).
           http://www.stat.rice.edu/~dobelman/textfiles/DistributionsHandbook.pdf

    .. versionadded:: 1.1.0

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        u1 = gamma.rvs(a=0.5, scale=2, size=size, random_state=random_state)
        return -np.log(u1)

    def _pdf(self, x):
        return np.exp(-0.5 * (x + np.exp(-x))) / np.sqrt(2 * np.pi)

    def _cdf(self, x):
        return sc.erfc(np.exp(-0.5 * x) / np.sqrt(2))

    def _sf(self, x):
        return sc.erf(np.exp(-0.5 * x) / np.sqrt(2))

    def _ppf(self, x):
        return -np.log(2 * sc.erfcinv(x) ** 2)

    def _stats(self):
        mu = np.log(2) + np.euler_gamma
        mu2 = np.pi ** 2 / 2
        g1 = 28 * np.sqrt(2) * sc.zeta(3) / np.pi ** 3
        g2 = 4.0
        return (mu, mu2, g1, g2)

    def _munp(self, n):
        if n == 1.0:
            return np.log(2) + np.euler_gamma
        elif n == 2.0:
            return np.pi ** 2 / 2 + (np.log(2) + np.euler_gamma) ** 2
        elif n == 3.0:
            tmp1 = 1.5 * np.pi ** 2 * (np.log(2) + np.euler_gamma)
            tmp2 = (np.log(2) + np.euler_gamma) ** 3
            tmp3 = 14 * sc.zeta(3)
            return tmp1 + tmp2 + tmp3
        elif n == 4.0:
            tmp1 = 4 * 14 * sc.zeta(3) * (np.log(2) + np.euler_gamma)
            tmp2 = 3 * np.pi ** 2 * (np.log(2) + np.euler_gamma) ** 2
            tmp3 = (np.log(2) + np.euler_gamma) ** 4
            tmp4 = 7 * np.pi ** 4 / 4
            return tmp1 + tmp2 + tmp3 + tmp4
        else:
            return self._mom1_sc(n)