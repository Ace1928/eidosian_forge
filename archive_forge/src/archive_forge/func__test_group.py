from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
import logging
def _test_group(pvalues, group_name, group, exact=True):
    """test if the objects in the group are different from the general set.

    The test is performed on the pvalues set (ad a pandas series) over
    the group specified via a fisher exact test.
    """
    from scipy.stats import fisher_exact, chi2_contingency
    totals = 1.0 * len(pvalues)
    total_significant = 1.0 * np.sum(pvalues)
    cross_index = [c for c in group if c in pvalues.index]
    missing = [c for c in group if c not in pvalues.index]
    if missing:
        s = 'the test is not well defined if the group has elements not presents in the significativity array. group name: {}, missing elements: {}'
        logging.warning(s.format(group_name, missing))
    group_total = 1.0 * len(cross_index)
    group_sign = 1.0 * len([c for c in cross_index if pvalues[c]])
    group_nonsign = 1.0 * (group_total - group_sign)
    extern_sign = 1.0 * (total_significant - group_sign)
    extern_nonsign = 1.0 * (totals - total_significant - group_nonsign)
    test = fisher_exact if exact else chi2_contingency
    table = [[extern_nonsign, extern_sign], [group_nonsign, group_sign]]
    pvalue = test(np.array(table))[1]
    part = (group_sign, group_nonsign, extern_sign, extern_nonsign)
    increase = np.log(totals * group_sign / (total_significant * group_total))
    return (pvalue, increase, part)