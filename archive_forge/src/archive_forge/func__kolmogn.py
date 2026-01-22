import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _kolmogn(n, x, cdf=True):
    """Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.

    Simard & L'Ecuyer (2011) [7].
    """
    if np.isnan(n):
        return n
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    t = n * x
    if t <= 1.0:
        if t <= 0.5:
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
        if n <= 140:
            prob = np.prod(np.arange(1, n + 1) * (1.0 / n) * (2 * t - 1))
        else:
            prob = np.exp(_log_nfactorial_div_n_pow_n(n) + n * np.log(2 * t - 1))
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
    if t >= n - 1:
        prob = 2 * (1.0 - x) ** n
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)
    if x >= 0.5:
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)
    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            prob = 2 * scipy.special.smirnov(n, x)
            return _clip_prob(prob)
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x ** 1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)