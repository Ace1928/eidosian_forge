import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
def anovadict(res):
    """update regression results dictionary with ANOVA specific statistics

    not checked for completeness
    """
    ad = {}
    ad.update(res.__dict__)
    anova_attr = ['df_model', 'df_resid', 'ess', 'ssr', 'uncentered_tss', 'mse_model', 'mse_resid', 'mse_total', 'fvalue', 'f_pvalue', 'rsquared']
    for key in anova_attr:
        ad[key] = getattr(res, key)
    ad['nobs'] = res.model.nobs
    ad['ssmwithmean'] = res.uncentered_tss - res.ssr
    return ad