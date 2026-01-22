import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def alexandergovern(*samples, nan_policy='propagate'):
    """Performs the Alexander Govern test.

    The Alexander-Govern approximation tests the equality of k independent
    means in the face of heterogeneity of variance. The test is applied to
    samples from two or more groups, possibly with differing sizes.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.  There must be at least
        two samples.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    res : AlexanderGovernResult
        An object with attributes:

        statistic : float
            The computed A statistic of the test.
        pvalue : float
            The associated p-value from the chi-squared distribution.

    Warns
    -----
    `~scipy.stats.ConstantInputWarning`
        Raised if an input is a constant array.  The statistic is not defined
        in this case, so ``np.nan`` is returned.

    See Also
    --------
    f_oneway : one-way ANOVA

    Notes
    -----
    The use of this test relies on several assumptions.

    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. Unlike `f_oneway`, this test does not assume on homoscedasticity,
       instead relaxing the assumption of equal variances.

    Input samples must be finite, one dimensional, and with size greater than
    one.

    References
    ----------
    .. [1] Alexander, Ralph A., and Diane M. Govern. "A New and Simpler
           Approximation for ANOVA under Variance Heterogeneity." Journal
           of Educational Statistics, vol. 19, no. 2, 1994, pp. 91-101.
           JSTOR, www.jstor.org/stable/1165140. Accessed 12 Sept. 2020.

    Examples
    --------
    >>> from scipy.stats import alexandergovern

    Here are some data on annual percentage rate of interest charged on
    new car loans at nine of the largest banks in four American cities
    taken from the National Institute of Standards and Technology's
    ANOVA dataset.

    We use `alexandergovern` to test the null hypothesis that all cities
    have the same mean APR against the alternative that the cities do not
    all have the same mean APR. We decide that a significance level of 5%
    is required to reject the null hypothesis in favor of the alternative.

    >>> atlanta = [13.75, 13.75, 13.5, 13.5, 13.0, 13.0, 13.0, 12.75, 12.5]
    >>> chicago = [14.25, 13.0, 12.75, 12.5, 12.5, 12.4, 12.3, 11.9, 11.9]
    >>> houston = [14.0, 14.0, 13.51, 13.5, 13.5, 13.25, 13.0, 12.5, 12.5]
    >>> memphis = [15.0, 14.0, 13.75, 13.59, 13.25, 12.97, 12.5, 12.25,
    ...           11.89]
    >>> alexandergovern(atlanta, chicago, houston, memphis)
    AlexanderGovernResult(statistic=4.65087071883494,
                          pvalue=0.19922132490385214)

    The p-value is 0.1992, indicating a nearly 20% chance of observing
    such an extreme value of the test statistic under the null hypothesis.
    This exceeds 5%, so we do not reject the null hypothesis in favor of
    the alternative.

    """
    samples = _alexandergovern_input_validation(samples, nan_policy)
    if np.any([(sample == sample[0]).all() for sample in samples]):
        msg = 'An input array is constant; the statistic is not defined.'
        warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)
        return AlexanderGovernResult(np.nan, np.nan)
    lengths = np.array([ma.count(sample) if nan_policy == 'omit' else len(sample) for sample in samples])
    means = np.array([np.mean(sample) for sample in samples])
    standard_errors = [np.std(sample, ddof=1) / np.sqrt(length) for sample, length in zip(samples, lengths)]
    inv_sq_se = 1 / np.square(standard_errors)
    weights = inv_sq_se / np.sum(inv_sq_se)
    var_w = np.sum(weights * means)
    t_stats = (means - var_w) / standard_errors
    v = lengths - 1
    a = v - 0.5
    b = 48 * a ** 2
    c = (a * np.log(1 + t_stats ** 2 / v)) ** 0.5
    z = c + (c ** 3 + 3 * c) / b - (4 * c ** 7 + 33 * c ** 5 + 240 * c ** 3 + 855 * c) / (b ** 2 * 10 + 8 * b * c ** 4 + 1000 * b)
    A = np.sum(np.square(z))
    p = distributions.chi2.sf(A, len(samples) - 1)
    return AlexanderGovernResult(A, p)