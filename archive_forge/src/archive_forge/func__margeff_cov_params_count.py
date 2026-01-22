from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _margeff_cov_params_count(model, cov_margins, params, exog, count_ind, method, J):
    """
    Returns the Jacobian for discrete regressors for use in margeff_cov_params.

    For discrete regressors the marginal effect is

    \\Delta F = F(XB) | d += 1 - F(XB) | d -= 1

    The row of the Jacobian for this variable is given by

    (f(XB)*X | d += 1 - f(XB)*X | d -= 1) / 2

    where F is the default prediction for the model.
    """
    for i in count_ind:
        exog0 = exog.copy()
        exog0[:, i] -= 1
        dfdb0 = model._derivative_predict(params, exog0, method)
        exog0[:, i] += 2
        dfdb1 = model._derivative_predict(params, exog0, method)
        dfdb = dfdb1 - dfdb0
        if dfdb.ndim >= 2:
            dfdb = dfdb.mean(0) / 2
        if J > 1:
            K = dfdb.shape[1] / (J - 1)
            cov_margins[i::K, :] = dfdb
        else:
            cov_margins[i, :len(dfdb)] = dfdb
    return cov_margins