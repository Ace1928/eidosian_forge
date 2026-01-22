from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
@_axis_nan_policy_factory(AnsariResult, n_samples=2)
def ansari(x, y, alternative='two-sided'):
    """Perform the Ansari-Bradley test for equal scale parameters.

    The Ansari-Bradley test ([1]_, [2]_) is a non-parametric test
    for the equality of the scale parameter of the distributions
    from which two samples were drawn. The null hypothesis states that
    the ratio of the scale of the distribution underlying `x` to the scale
    of the distribution underlying `y` is 1.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the ratio of scales is not equal to 1.
        * 'less': the ratio of scales is less than 1.
        * 'greater': the ratio of scales is greater than 1.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The Ansari-Bradley test statistic.
    pvalue : float
        The p-value of the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    mood : A non-parametric test for the equality of two scale parameters

    Notes
    -----
    The p-value given is exact when the sample sizes are both less than
    55 and there are no ties, otherwise a normal approximation for the
    p-value is used.

    References
    ----------
    .. [1] Ansari, A. R. and Bradley, R. A. (1960) Rank-sum tests for
           dispersions, Annals of Mathematical Statistics, 31, 1174-1189.
    .. [2] Sprent, Peter and N.C. Smeeton.  Applied nonparametric
           statistical methods.  3rd ed. Chapman and Hall/CRC. 2001.
           Section 5.8.2.
    .. [3] Nathaniel E. Helwig "Nonparametric Dispersion and Equality
           Tests" at http://users.stat.umn.edu/~helwig/notes/npde-Notes.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ansari
    >>> rng = np.random.default_rng()

    For these examples, we'll create three random data sets.  The first
    two, with sizes 35 and 25, are drawn from a normal distribution with
    mean 0 and standard deviation 2.  The third data set has size 25 and
    is drawn from a normal distribution with standard deviation 1.25.

    >>> x1 = rng.normal(loc=0, scale=2, size=35)
    >>> x2 = rng.normal(loc=0, scale=2, size=25)
    >>> x3 = rng.normal(loc=0, scale=1.25, size=25)

    First we apply `ansari` to `x1` and `x2`.  These samples are drawn
    from the same distribution, so we expect the Ansari-Bradley test
    should not lead us to conclude that the scales of the distributions
    are different.

    >>> ansari(x1, x2)
    AnsariResult(statistic=541.0, pvalue=0.9762532927399098)

    With a p-value close to 1, we cannot conclude that there is a
    significant difference in the scales (as expected).

    Now apply the test to `x1` and `x3`:

    >>> ansari(x1, x3)
    AnsariResult(statistic=425.0, pvalue=0.0003087020407974518)

    The probability of observing such an extreme value of the statistic
    under the null hypothesis of equal scales is only 0.03087%. We take this
    as evidence against the null hypothesis in favor of the alternative:
    the scales of the distributions from which the samples were drawn
    are not equal.

    We can use the `alternative` parameter to perform a one-tailed test.
    In the above example, the scale of `x1` is greater than `x3` and so
    the ratio of scales of `x1` and `x3` is greater than 1. This means
    that the p-value when ``alternative='greater'`` should be near 0 and
    hence we should be able to reject the null hypothesis:

    >>> ansari(x1, x3, alternative='greater')
    AnsariResult(statistic=425.0, pvalue=0.0001543510203987259)

    As we can see, the p-value is indeed quite low. Use of
    ``alternative='less'`` should thus yield a large p-value:

    >>> ansari(x1, x3, alternative='less')
    AnsariResult(statistic=425.0, pvalue=0.9998643258449039)

    """
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("'alternative' must be 'two-sided', 'greater', or 'less'.")
    x, y = (asarray(x), asarray(y))
    n = len(x)
    m = len(y)
    if m < 1:
        raise ValueError('Not enough other observations.')
    if n < 1:
        raise ValueError('Not enough test observations.')
    N = m + n
    xy = r_[x, y]
    rank = _stats_py.rankdata(xy)
    symrank = amin(array((rank, N - rank + 1)), 0)
    AB = np.sum(symrank[:n], axis=0)
    uxy = unique(xy)
    repeats = len(uxy) != len(xy)
    exact = m < 55 and n < 55 and (not repeats)
    if repeats and (m < 55 or n < 55):
        warnings.warn('Ties preclude use of exact statistic.', stacklevel=2)
    if exact:
        if alternative == 'two-sided':
            pval = 2.0 * np.minimum(_abw_state.cdf(AB, n, m), _abw_state.sf(AB, n, m))
        elif alternative == 'greater':
            pval = _abw_state.cdf(AB, n, m)
        else:
            pval = _abw_state.sf(AB, n, m)
        return AnsariResult(AB, min(1.0, pval))
    if N % 2:
        mnAB = n * (N + 1.0) ** 2 / 4.0 / N
        varAB = n * m * (N + 1.0) * (3 + N ** 2) / (48.0 * N ** 2)
    else:
        mnAB = n * (N + 2.0) / 4.0
        varAB = m * n * (N + 2) * (N - 2.0) / 48 / (N - 1.0)
    if repeats:
        fac = np.sum(symrank ** 2, axis=0)
        if N % 2:
            varAB = m * n * (16 * N * fac - (N + 1) ** 4) / (16.0 * N ** 2 * (N - 1))
        else:
            varAB = m * n * (16 * fac - N * (N + 2) ** 2) / (16.0 * N * (N - 1))
    z = (mnAB - AB) / sqrt(varAB)
    z, pval = _normtest_finish(z, alternative)
    return AnsariResult(AB, pval)