from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def tukey_hsd(*args):
    """Perform Tukey's HSD test for equality of means over multiple treatments.

    Tukey's honestly significant difference (HSD) test performs pairwise
    comparison of means for a set of samples. Whereas ANOVA (e.g. `f_oneway`)
    assesses whether the true means underlying each sample are identical,
    Tukey's HSD is a post hoc test used to compare the mean of each sample
    to the mean of each other sample.

    The null hypothesis is that the distributions underlying the samples all
    have the same mean. The test statistic, which is computed for every
    possible pairing of samples, is simply the difference between the sample
    means. For each pair, the p-value is the probability under the null
    hypothesis (and other assumptions; see notes) of observing such an extreme
    value of the statistic, considering that many pairwise comparisons are
    being performed. Confidence intervals for the difference between each pair
    of means are also available.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group. There must be at least
        two arguments.

    Returns
    -------
    result : `~scipy.stats._result_classes.TukeyHSDResult` instance
        The return value is an object with the following attributes:

        statistic : float ndarray
            The computed statistic of the test for each comparison. The element
            at index ``(i, j)`` is the statistic for the comparison between
            groups ``i`` and ``j``.
        pvalue : float ndarray
            The computed p-value of the test for each comparison. The element
            at index ``(i, j)`` is the p-value for the comparison between
            groups ``i`` and ``j``.

        The object has the following methods:

        confidence_interval(confidence_level=0.95):
            Compute the confidence interval for the specified confidence level.

    See Also
    --------
    dunnett : performs comparison of means against a control group.

    Notes
    -----
    The use of this test relies on several assumptions.

    1. The observations are independent within and among groups.
    2. The observations within each group are normally distributed.
    3. The distributions from which the samples are drawn have the same finite
       variance.

    The original formulation of the test was for samples of equal size [6]_.
    In case of unequal sample sizes, the test uses the Tukey-Kramer method
    [4]_.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1. Tukey's
           Method."
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
           28 November 2020.
    .. [2] Abdi, Herve & Williams, Lynne. (2021). "Tukey's Honestly Significant
           Difference (HSD) Test."
           https://personal.utdallas.edu/~herve/abdi-HSD2010-pretty.pdf
    .. [3] "One-Way ANOVA Using SAS PROC ANOVA & PROC GLM." SAS
           Tutorials, 2007, www.stattutorials.com/SAS/TUTORIAL-PROC-GLM.htm.
    .. [4] Kramer, Clyde Young. "Extension of Multiple Range Tests to Group
           Means with Unequal Numbers of Replications." Biometrics, vol. 12,
           no. 3, 1956, pp. 307-310. JSTOR, www.jstor.org/stable/3001469.
           Accessed 25 May 2021.
    .. [5] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.3.3.
           The ANOVA table and tests of hypotheses about means"
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm,
           2 June 2021.
    .. [6] Tukey, John W. "Comparing Individual Means in the Analysis of
           Variance." Biometrics, vol. 5, no. 2, 1949, pp. 99-114. JSTOR,
           www.jstor.org/stable/3001913. Accessed 14 June 2021.


    Examples
    --------
    Here are some data comparing the time to relief of three brands of
    headache medicine, reported in minutes. Data adapted from [3]_.

    >>> import numpy as np
    >>> from scipy.stats import tukey_hsd
    >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
    >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
    >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]

    We would like to see if the means between any of the groups are
    significantly different. First, visually examine a box and whisker plot.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.boxplot([group0, group1, group2])
    >>> ax.set_xticklabels(["group0", "group1", "group2"]) # doctest: +SKIP
    >>> ax.set_ylabel("mean") # doctest: +SKIP
    >>> plt.show()

    From the box and whisker plot, we can see overlap in the interquartile
    ranges group 1 to group 2 and group 3, but we can apply the ``tukey_hsd``
    test to determine if the difference between means is significant. We
    set a significance level of .05 to reject the null hypothesis.

    >>> res = tukey_hsd(group0, group1, group2)
    >>> print(res)
    Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
    Comparison  Statistic  p-value   Lower CI   Upper CI
    (0 - 1)     -4.600      0.014     -8.249     -0.951
    (0 - 2)     -0.260      0.980     -3.909      3.389
    (1 - 0)      4.600      0.014      0.951      8.249
    (1 - 2)      4.340      0.020      0.691      7.989
    (2 - 0)      0.260      0.980     -3.389      3.909
    (2 - 1)     -4.340      0.020     -7.989     -0.691

    The null hypothesis is that each group has the same mean. The p-value for
    comparisons between ``group0`` and ``group1`` as well as ``group1`` and
    ``group2`` do not exceed .05, so we reject the null hypothesis that they
    have the same means. The p-value of the comparison between ``group0``
    and ``group2`` exceeds .05, so we accept the null hypothesis that there
    is not a significant difference between their means.

    We can also compute the confidence interval associated with our chosen
    confidence level.

    >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
    >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
    >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
    >>> result = tukey_hsd(group0, group1, group2)
    >>> conf = res.confidence_interval(confidence_level=.99)
    >>> for ((i, j), l) in np.ndenumerate(conf.low):
    ...     # filter out self comparisons
    ...     if i != j:
    ...         h = conf.high[i,j]
    ...         print(f"({i} - {j}) {l:>6.3f} {h:>6.3f}")
    (0 - 1) -9.480  0.280
    (0 - 2) -5.140  4.620
    (1 - 0) -0.280  9.480
    (1 - 2) -0.540  9.220
    (2 - 0) -4.620  5.140
    (2 - 1) -9.220  0.540
    """
    args = _tukey_hsd_iv(args)
    ntreatments = len(args)
    means = np.asarray([np.mean(arg) for arg in args])
    nsamples_treatments = np.asarray([a.size for a in args])
    nobs = np.sum(nsamples_treatments)
    mse = np.sum([np.var(arg, ddof=1) for arg in args] * (nsamples_treatments - 1)) / (nobs - ntreatments)
    if np.unique(nsamples_treatments).size == 1:
        normalize = 2 / nsamples_treatments[0]
    else:
        normalize = 1 / nsamples_treatments + 1 / nsamples_treatments[None].T
    stand_err = np.sqrt(normalize * mse / 2)
    mean_differences = means[None].T - means
    t_stat = np.abs(mean_differences) / stand_err
    params = (t_stat, ntreatments, nobs - ntreatments)
    pvalues = distributions.studentized_range.sf(*params)
    return TukeyHSDResult(mean_differences, pvalues, ntreatments, nobs, stand_err)