from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
class randint_gen(rv_discrete):
    """A uniform discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `randint` is:

    .. math::

        f(k) = \\frac{1}{\\texttt{high} - \\texttt{low}}

    for :math:`k \\in \\{\\texttt{low}, \\dots, \\texttt{high} - 1\\}`.

    `randint` takes :math:`\\texttt{low}` and :math:`\\texttt{high}` as shape
    parameters.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import randint
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> low, high = 7, 31
    >>> mean, var, skew, kurt = randint.stats(low, high, moments='mvsk')

    Display the probability mass function (``pmf``):

    >>> x = np.arange(low - 5, high + 5)
    >>> ax.plot(x, randint.pmf(x, low, high), 'bo', ms=8, label='randint pmf')
    >>> ax.vlines(x, 0, randint.pmf(x, low, high), colors='b', lw=5, alpha=0.5)
    
    Alternatively, the distribution object can be called (as a function) to 
    fix the shape and location. This returns a "frozen" RV object holding the
    given parameters fixed.

    Freeze the distribution and display the frozen ``pmf``:

    >>> rv = randint(low, high)
    >>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-',
    ...           lw=1, label='frozen pmf')
    >>> ax.legend(loc='lower center')
    >>> plt.show()
    
    Check the relationship between the cumulative distribution function
    (``cdf``) and its inverse, the percent point function (``ppf``):

    >>> q = np.arange(low, high)
    >>> p = randint.cdf(q, low, high)
    >>> np.allclose(q, randint.ppf(p, low, high))
    True

    Generate random numbers:

    >>> r = randint.rvs(low, high, size=1000)

    """

    def _shape_info(self):
        return [_ShapeInfo('low', True, (-np.inf, np.inf), (False, False)), _ShapeInfo('high', True, (-np.inf, np.inf), (False, False))]

    def _argcheck(self, low, high):
        return (high > low) & _isintegral(low) & _isintegral(high)

    def _get_support(self, low, high):
        return (low, high - 1)

    def _pmf(self, k, low, high):
        p = np.ones_like(k) / (high - low)
        return np.where((k >= low) & (k < high), p, 0.0)

    def _cdf(self, x, low, high):
        k = floor(x)
        return (k - low + 1.0) / (high - low)

    def _ppf(self, q, low, high):
        vals = ceil(q * (high - low) + low) - 1
        vals1 = (vals - 1).clip(low, high)
        temp = self._cdf(vals1, low, high)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, low, high):
        m2, m1 = (np.asarray(high), np.asarray(low))
        mu = (m2 + m1 - 1.0) / 2
        d = m2 - m1
        var = (d * d - 1) / 12.0
        g1 = 0.0
        g2 = -6.0 / 5.0 * (d * d + 1.0) / (d * d - 1.0)
        return (mu, var, g1, g2)

    def _rvs(self, low, high, size=None, random_state=None):
        """An array of *size* random integers >= ``low`` and < ``high``."""
        if np.asarray(low).size == 1 and np.asarray(high).size == 1:
            return rng_integers(random_state, low, high, size=size)
        if size is not None:
            low = np.broadcast_to(low, size)
            high = np.broadcast_to(high, size)
        randint = np.vectorize(partial(rng_integers, random_state), otypes=[np.dtype(int)])
        return randint(low, high)

    def _entropy(self, low, high):
        return log(high - low)