import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def wellek_to_f2(eps, n_groups):
    """Convert Wellek's effect size (sqrt) to Cohen's f-squared

    This computes the following effect size :

       f2 = 1 / n_groups * eps**2

    Parameters
    ----------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    f2 : effect size Cohen's f-squared

    """
    f2 = 1 / n_groups * eps ** 2
    return f2