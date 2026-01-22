from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def anova2_lm_single(model, design_info, n_rows, test, pr_test, robust):
    """
    Anova type II table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.

    Type II
    Sum of Squares compares marginal contribution of terms. Thus, it is
    not particularly useful for models with significant interaction terms.
    """
    terms_info = design_info.terms[:]
    terms_info = _remove_intercept_patsy(terms_info)
    names = ['sum_sq', 'df', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 4)), columns=names)
    cov = _get_covariance(model, None)
    robust_cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        cols = design_info.slice(term)
        L1 = lrange(cols.start, cols.stop)
        L2 = []
        term_set = set(term.factors)
        for t in terms_info:
            other_set = set(t.factors)
            if term_set.issubset(other_set) and (not term_set == other_set):
                col = design_info.slice(t)
                L1.extend(lrange(col.start, col.stop))
                L2.extend(lrange(col.start, col.stop))
        L1 = np.eye(model.model.exog.shape[1])[L1]
        L2 = np.eye(model.model.exog.shape[1])[L2]
        if L2.size:
            LVL = np.dot(np.dot(L1, robust_cov), L2.T)
            from scipy import linalg
            orth_compl, _ = linalg.qr(LVL)
            r = L1.shape[0] - L2.shape[0]
            L12 = np.dot(orth_compl[:, -r:].T, L1)
        else:
            L12 = L1
            r = L1.shape[0]
        if test == 'F':
            f = model.f_test(L12, cov_p=robust_cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue
        table.loc[table.index[i], 'df'] = r
        col_order.append(cols.start)
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    table = table.iloc[np.argsort(col_order + [model.model.exog.shape[1] + 1])]
    ssr = table[test] * table['df'] * model.ssr / model.df_resid
    table['sum_sq'] = ssr
    table.loc['Residual', ['sum_sq', 'df', test, pr_test]] = (model.ssr, model.df_resid, np.nan, np.nan)
    return table