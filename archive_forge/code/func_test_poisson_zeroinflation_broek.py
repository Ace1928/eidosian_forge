import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def test_poisson_zeroinflation_broek(results_poisson):
    """score test for zero modification in Poisson, special case

    This assumes that the Poisson model has a constant and that
    the zero modification probability is constant.

    This is a special case of test_poisson_zeroinflation derived by
    van den Broek 1995.

    The test reports two sided and one sided alternatives based on
    the normal distribution of the test statistic.

    References
    ----------
    .. [1] Broek, Jan van den. 1995. “A Score Test for Zero Inflation in a
           Poisson Distribution.” Biometrics 51 (2): 738–43.
           https://doi.org/10.2307/2532959.

    """
    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)
    endog = results_poisson.model.endog
    score = (((endog == 0) - prob_zero) / prob_zero).sum()
    var_score = ((1 - prob_zero) / prob_zero).sum() - endog.sum()
    statistic = score / np.sqrt(var_score)
    pvalue_two = 2 * stats.norm.sf(np.abs(statistic))
    pvalue_upp = stats.norm.sf(statistic)
    pvalue_low = stats.norm.cdf(statistic)
    res = HolderTuple(statistic=statistic, pvalue=pvalue_two, pvalue_smaller=pvalue_upp, pvalue_larger=pvalue_low, chi2=statistic ** 2, pvalue_chi2=stats.chi2.sf(statistic ** 2, 1), df_chi2=1, distribution='normal')
    return res