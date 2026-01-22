import numpy as np
from scipy import special
from statsmodels.stats.base import Holder
def _noncentrality_chisquare(chi2_stat, df, alpha=0.05):
    """noncentrality parameter for chi-square statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    chi2_stat : float
        Chisquare-statistic, for example from a hypothesis test
    df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of
        Noncentrality Parameters.”
        Canadian Journal of Statistics 21 (1): 45–57.
        https://doi.org/10.2307/3315657.

    .. [2] Li, Qizhai, Junjian Zhang, and Shuai Dai. 2009. “On Estimating the
        Non-Centrality Parameter of a Chi-Squared Distribution.”
        Statistics & Probability Letters 79 (1): 98–104.
        https://doi.org/10.1016/j.spl.2008.07.025.

    """
    alpha_half = alpha / 2
    nc_umvue = chi2_stat - df
    nc = np.maximum(nc_umvue, 0)
    nc_lzd = np.maximum(nc_umvue, chi2_stat / (df + 1))
    nc_krs = np.maximum(nc_umvue, chi2_stat * 2 / (df + 2))
    nc_median = special.chndtrinc(chi2_stat, df, 0.5)
    ci = special.chndtrinc(chi2_stat, df, [1 - alpha_half, alpha_half])
    res = Holder(nc=nc, confint=ci, nc_umvue=nc_umvue, nc_lzd=nc_lzd, nc_krs=nc_krs, nc_median=nc_median, name='Noncentrality for chisquare-distributed random variable')
    return res