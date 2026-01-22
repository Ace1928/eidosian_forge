import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def _fstat2effectsize(f_stat, df):
    """Compute anova effect size from F-statistic

    This might be combined with convert_effectsize_fsqu

    Parameters
    ----------
    f_stat : array_like
        Test statistic of an F-test
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where
         - df1 : numerator degrees of freedom, number of constraints
         - df2 : denominator degrees of freedom, df_resid

    Returns
    -------
    res : Holder instance
        This instance contains effect size measures f2, eta2, omega2 and eps2
        as attributes.

    Notes
    -----
    This uses the following definitions:

    - f2 = f_stat * df1 / df2
    - eta2 = f2 / (f2 + 1)
    - omega2 = (f2 - df1 / df2) / (f2 + 2)
    - eps2 = (f2 - df1 / df2) / (f2 + 1)

    This differs from effect size measures in other function which define
    ``f2 = f_stat * df1 / nobs``
    or an equivalent expression for power computation. The noncentrality
    index for the hypothesis test is in those cases given by
    ``nc = f_stat * df1``.

    Currently omega2 and eps2 are computed in two different ways. Those
    values agree for regular cases but can show different behavior in corner
    cases (e.g. zero division).

    """
    df1, df2 = df
    f2 = f_stat * df1 / df2
    eta2 = f2 / (f2 + 1)
    omega2_ = (f_stat - 1) / (f_stat + (df2 + 1) / df1)
    omega2 = (f2 - df1 / df2) / (f2 + 1 + 1 / df2)
    eps2_ = (f_stat - 1) / (f_stat + df2 / df1)
    eps2 = (f2 - df1 / df2) / (f2 + 1)
    return Holder(f2=f2, eta2=eta2, omega2=omega2, eps2=eps2, eps2_=eps2_, omega2_=omega2_)