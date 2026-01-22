import numpy as np
from scipy import special
from statsmodels.stats.base import Holder
def _noncentrality_f(f_stat, df1, df2, alpha=0.05):
    """noncentrality parameter for f statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    fstat : float
        f-statistic, for example from a hypothesis test
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
       Noncentrality Parameters.” Canadian Journal of Statistics 21 (1): 45–57.
       https://doi.org/10.2307/3315657.
    """
    alpha_half = alpha / 2
    x_s = f_stat * df1 / df2
    nc_umvue = (df2 - 2) * x_s - df1
    nc = np.maximum(nc_umvue, 0)
    nc_krs = np.maximum(nc_umvue, x_s * 2 * (df2 - 1) / (df1 + 2))
    nc_median = special.ncfdtrinc(df1, df2, 0.5, f_stat)
    ci = special.ncfdtrinc(df1, df2, [1 - alpha_half, alpha_half], f_stat)
    res = Holder(nc=nc, confint=ci, nc_umvue=nc_umvue, nc_krs=nc_krs, nc_median=nc_median, name='Noncentrality for F-distributed random variable')
    return res