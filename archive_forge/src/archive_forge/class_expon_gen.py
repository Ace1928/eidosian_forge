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
class expon_gen(rv_continuous):
    """An exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `expon` is:

    .. math::

        f(x) = \\exp(-x)

    for :math:`x \\ge 0`.

    %(after_notes)s

    A common parameterization for `expon` is in terms of the rate parameter
    ``lambda``, such that ``pdf = lambda * exp(-lambda * x)``. This
    parameterization corresponds to using ``scale = 1 / lambda``.

    The exponential distribution is a special case of the gamma
    distributions, with gamma shape parameter ``a = 1``.

    %(example)s

    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.standard_exponential(size)

    def _pdf(self, x):
        return np.exp(-x)

    def _logpdf(self, x):
        return -x

    def _cdf(self, x):
        return -sc.expm1(-x)

    def _ppf(self, q):
        return -sc.log1p(-q)

    def _sf(self, x):
        return np.exp(-x)

    def _logsf(self, x):
        return -x

    def _isf(self, q):
        return -np.log(q)

    def _stats(self):
        return (1.0, 1.0, 2.0, 6.0)

    def _entropy(self):
        return 1.0

    @_call_super_mom
    @replace_notes_in_docstring(rv_continuous, notes="        When `method='MLE'`,\n        this function uses explicit formulas for the maximum likelihood\n        estimation of the exponential distribution parameters, so the\n        `optimizer`, `loc` and `scale` keyword arguments are\n        ignored.\n\n")
    def fit(self, data, *args, **kwds):
        if len(args) > 0:
            raise TypeError('Too many arguments.')
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)
        _remove_optimizer_parameters(kwds)
        if floc is not None and fscale is not None:
            raise ValueError('All parameters fixed. There is nothing to optimize.')
        data = np.asarray(data)
        if not np.isfinite(data).all():
            raise ValueError('The data contains non-finite values.')
        data_min = data.min()
        if floc is None:
            loc = data_min
        else:
            loc = floc
            if data_min < loc:
                raise FitDataError('expon', lower=floc, upper=np.inf)
        if fscale is None:
            scale = data.mean() - loc
        else:
            scale = fscale
        return (float(loc), float(scale))