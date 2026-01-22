from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def anova3_lm_single(model, design_info, n_rows, test, pr_test, robust):
    n_rows += _has_intercept(design_info)
    terms_info = design_info.terms
    names = ['sum_sq', 'df', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 4)), columns=names)
    cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        cols = design_info.slice(term)
        L1 = np.eye(model.model.exog.shape[1])[cols]
        L12 = L1
        r = L1.shape[0]
        if test == 'F':
            f = model.f_test(L12, cov_p=cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue
        table.loc[table.index[i], 'df'] = r
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    ssr = table[test] * table['df'] * model.ssr / model.df_resid
    table['sum_sq'] = ssr
    table.loc['Residual', ['sum_sq', 'df', test, pr_test]] = (model.ssr, model.df_resid, np.nan, np.nan)
    return table