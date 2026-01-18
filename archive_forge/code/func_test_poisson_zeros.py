import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def test_poisson_zeros(results):
    """Test for excess zeros in Poisson regression model.

    The test is implemented following Tang and Tang [1]_ equ. (12) which is
    based on the test derived in He et al 2019 [2]_.

    References
    ----------

    .. [1] Tang, Yi, and Wan Tang. 2018. “Testing Modified Zeros for Poisson
           Regression Models:” Statistical Methods in Medical Research,
           September. https://doi.org/10.1177/0962280218796253.

    .. [2] He, Hua, Hui Zhang, Peng Ye, and Wan Tang. 2019. “A Test of Inflated
           Zeros for Poisson Regression Models.” Statistical Methods in
           Medical Research 28 (4): 1157–69.
           https://doi.org/10.1177/0962280217749991.

    """
    x = results.model.exog
    mean = results.predict()
    prob0 = np.exp(-mean)
    counts = (results.model.endog == 0).astype(int)
    diff = counts.sum() - prob0.sum()
    var1 = prob0 @ (1 - prob0)
    pm = prob0 * mean
    c = np.linalg.inv(x.T * mean @ x)
    pmx = pm @ x
    var2 = pmx @ c @ pmx
    var = var1 - var2
    statistic = diff / np.sqrt(var)
    pvalue_two = 2 * stats.norm.sf(np.abs(statistic))
    pvalue_upp = stats.norm.sf(statistic)
    pvalue_low = stats.norm.cdf(statistic)
    res = HolderTuple(statistic=statistic, pvalue=pvalue_two, pvalue_smaller=pvalue_upp, pvalue_larger=pvalue_low, chi2=statistic ** 2, pvalue_chi2=stats.chi2.sf(statistic ** 2, 1), df_chi2=1, distribution='normal')
    return res