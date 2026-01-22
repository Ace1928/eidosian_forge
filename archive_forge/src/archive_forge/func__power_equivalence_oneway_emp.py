import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, df, alpha=0.05):
    """Empirical power of oneway equivalence test

    This only returns post-hoc, empirical power.

    Warning: eps is currently effect size margin as defined as in Wellek, and
    not the signal to noise ratio (Cohen's f family).

    Parameters
    ----------
    f_stat : float
        F-statistic from oneway anova, used to compute empirical effect size
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    eps : float
        Equivalence margin in terms of effect size given by Wellek's psi.
    df : tuple
        Degrees of freedom for F distribution.
    alpha : float in (0, 1)
        Significance level for the hypothesis test.

    Returns
    -------
    pow : float
        Ex-post, post-hoc or empirical power at f-statistic of the equivalence
        test.
    """
    res = equivalence_oneway_generic(f_stat, n_groups, nobs, eps, df, alpha=alpha, margin_type='wellek')
    nobs_mean = nobs.sum() / n_groups
    fn = f_stat
    esn = fn * (n_groups - 1) / nobs_mean
    pow_ = ncf_cdf(res.crit_f, df[0], df[1], nobs_mean * esn)
    return pow_