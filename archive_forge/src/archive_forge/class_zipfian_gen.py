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
class zipfian_gen(rv_discrete):
    """A Zipfian discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipf

    Notes
    -----
    The probability mass function for `zipfian` is:

    .. math::

        f(k, a, n) = \\frac{1}{H_{n,a} k^a}

    for :math:`k \\in \\{1, 2, \\dots, n-1, n\\}`, :math:`a \\ge 0`,
    :math:`n \\in \\{1, 2, 3, \\dots\\}`.

    `zipfian` takes :math:`a` and :math:`n` as shape parameters.
    :math:`H_{n,a}` is the :math:`n`:sup:`th` generalized harmonic
    number of order :math:`a`.

    The Zipfian distribution reduces to the Zipf (zeta) distribution as
    :math:`n \\rightarrow \\infty`.

    %(after_notes)s

    References
    ----------
    .. [1] "Zipf's Law", Wikipedia, https://en.wikipedia.org/wiki/Zipf's_law
    .. [2] Larry Leemis, "Zipf Distribution", Univariate Distribution
           Relationships. http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf

    %(example)s

    Confirm that `zipfian` reduces to `zipf` for large `n`, `a > 1`.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipfian.pmf(k, a=3.5, n=10000000), zipf.pmf(k, a=3.5))
    True

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (True, False)), _ShapeInfo('n', True, (0, np.inf), (False, False))]

    def _argcheck(self, a, n):
        return (a >= 0) & (n > 0) & (n == np.asarray(n, dtype=int))

    def _get_support(self, a, n):
        return (1, n)

    def _pmf(self, k, a, n):
        return 1.0 / _gen_harmonic(n, a) / k ** a

    def _cdf(self, k, a, n):
        return _gen_harmonic(k, a) / _gen_harmonic(n, a)

    def _sf(self, k, a, n):
        k = k + 1
        return (k ** a * (_gen_harmonic(n, a) - _gen_harmonic(k, a)) + 1) / (k ** a * _gen_harmonic(n, a))

    def _stats(self, a, n):
        Hna = _gen_harmonic(n, a)
        Hna1 = _gen_harmonic(n, a - 1)
        Hna2 = _gen_harmonic(n, a - 2)
        Hna3 = _gen_harmonic(n, a - 3)
        Hna4 = _gen_harmonic(n, a - 4)
        mu1 = Hna1 / Hna
        mu2n = Hna2 * Hna - Hna1 ** 2
        mu2d = Hna ** 2
        mu2 = mu2n / mu2d
        g1 = (Hna3 / Hna - 3 * Hna1 * Hna2 / Hna ** 2 + 2 * Hna1 ** 3 / Hna ** 3) / mu2 ** (3 / 2)
        g2 = (Hna ** 3 * Hna4 - 4 * Hna ** 2 * Hna1 * Hna3 + 6 * Hna * Hna1 ** 2 * Hna2 - 3 * Hna1 ** 4) / mu2n ** 2
        g2 -= 3
        return (mu1, mu2, g1, g2)