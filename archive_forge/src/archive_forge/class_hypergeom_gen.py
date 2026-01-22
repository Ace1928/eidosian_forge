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
class hypergeom_gen(rv_discrete):
    """A hypergeometric discrete random variable.

    The hypergeometric distribution models drawing objects from a bin.
    `M` is the total number of objects, `n` is total number of Type I objects.
    The random variate represents the number of Type I objects in `N` drawn
    without replacement from the total population.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
    universally accepted.  See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: p(k, M, n, N) = \\frac{\\binom{n}{k} \\binom{M - n}{N - k}}
                                   {\\binom{M}{N}}

    for :math:`k \\in [\\max(0, N - M + n), \\min(n, N)]`, where the binomial
    coefficients are defined as,

    .. math:: \\binom{n}{k} \\equiv \\frac{n!}{k! (n - k)!}.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import hypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if
    we want to know the probability of finding a given number of dogs if we
    choose at random 12 of the 20 animals, we can initialize a frozen
    distribution and plot the probability mass function:

    >>> [M, n, N] = [20, 7, 12]
    >>> rv = hypergeom(M, n, N)
    >>> x = np.arange(0, n+1)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group of chosen animals')
    >>> ax.set_ylabel('hypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `hypergeom`
    methods directly.  To for example obtain the cumulative distribution
    function, use:

    >>> prb = hypergeom.cdf(x, M, n, N)

    And to generate random numbers:

    >>> R = hypergeom.rvs(M, n, N, size=10)

    See Also
    --------
    nhypergeom, binom, nbinom

    """

    def _shape_info(self):
        return [_ShapeInfo('M', True, (0, np.inf), (True, False)), _ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('N', True, (0, np.inf), (True, False))]

    def _rvs(self, M, n, N, size=None, random_state=None):
        return random_state.hypergeometric(n, M - n, N, size=size)

    def _get_support(self, M, n, N):
        return (np.maximum(N - (M - n), 0), np.minimum(n, N))

    def _argcheck(self, M, n, N):
        cond = (M > 0) & (n >= 0) & (N >= 0)
        cond &= (n <= M) & (N <= M)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(N)
        return cond

    def _logpmf(self, k, M, n, N):
        tot, good = (M, n)
        bad = tot - good
        result = betaln(good + 1, 1) + betaln(bad + 1, 1) + betaln(tot - N + 1, N + 1) - betaln(k + 1, good - k + 1) - betaln(N - k + 1, bad - N + k + 1) - betaln(tot + 1, 1)
        return result

    def _pmf(self, k, M, n, N):
        return _boost._hypergeom_pdf(k, n, N, M)

    def _cdf(self, k, M, n, N):
        return _boost._hypergeom_cdf(k, n, N, M)

    def _stats(self, M, n, N):
        M, n, N = (1.0 * M, 1.0 * n, 1.0 * N)
        m = M - n
        g2 = M * (M + 1) - 6.0 * N * (M - N) - 6.0 * n * m
        g2 *= (M - 1) * M * M
        g2 += 6.0 * n * N * (M - N) * m * (5.0 * M - 6)
        g2 /= n * N * (M - N) * m * (M - 2.0) * (M - 3.0)
        return (_boost._hypergeom_mean(n, N, M), _boost._hypergeom_variance(n, N, M), _boost._hypergeom_skewness(n, N, M), g2)

    def _entropy(self, M, n, N):
        k = np.r_[N - (M - n):min(n, N) + 1]
        vals = self.pmf(k, M, n, N)
        return np.sum(entr(vals), axis=0)

    def _sf(self, k, M, n, N):
        return _boost._hypergeom_sf(k, n, N, M)

    def _logsf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) < (good - 0.5) * (draw - 0.5):
                res.append(log1p(-exp(self.logcdf(quant, tot, good, draw))))
            else:
                k2 = np.arange(quant + 1, draw + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)

    def _logcdf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) > (good - 0.5) * (draw - 0.5):
                res.append(log1p(-exp(self.logsf(quant, tot, good, draw))))
            else:
                k2 = np.arange(0, quant + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)