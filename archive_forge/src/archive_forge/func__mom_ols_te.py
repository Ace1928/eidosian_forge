import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def _mom_ols_te(tm, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    first moment is ATE
    second moment is POM0  (control)

    """
    w = tind / prob + (1 - tind) / (1 - prob)
    treat_ind = np.column_stack((tind, np.ones(len(tind))))
    mom = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind
    return mom