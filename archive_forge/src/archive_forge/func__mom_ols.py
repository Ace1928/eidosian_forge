import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def _mom_ols(params, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    moment conditions are POM0 and POM1

    """
    w = tind / prob + (1 - tind) / (1 - prob)
    treat_ind = np.column_stack((1 - tind, tind))
    mom = (w * (endog - treat_ind.dot(params)))[:, None] * treat_ind
    return mom