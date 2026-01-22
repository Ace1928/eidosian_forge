from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def anova_single(model, **kwargs):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model
    typ : int or str {1,2,3} or {"I","II","III"}
        Type of sum of squares to use.

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    test = kwargs.get('test', 'F')
    scale = kwargs.get('scale', None)
    typ = kwargs.get('typ', 1)
    robust = kwargs.get('robust', None)
    if robust:
        robust = robust.lower()
    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]
    response_name = model.model.endog_names
    design_info = model.model.data.design_info
    exog_names = model.model.exog_names
    n_rows = len(design_info.terms) - _has_intercept(design_info) + 1
    pr_test = 'PR(>%s)' % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 5)), columns=names)
    if typ in [1, 'I']:
        return anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test, pr_test, robust)
    elif typ in [2, 'II']:
        return anova2_lm_single(model, design_info, n_rows, test, pr_test, robust)
    elif typ in [3, 'III']:
        return anova3_lm_single(model, design_info, n_rows, test, pr_test, robust)
    elif typ in [4, 'IV']:
        raise NotImplementedError('Type IV not yet implemented')
    else:
        raise ValueError('Type %s not understood' % str(typ))