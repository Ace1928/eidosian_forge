import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def _mom_atm(params, endog, tind, prob, weighted=True):
    """moment conditions for average treatment means (POM)

    moment conditions are POM0 and POM1
    """
    w1 = tind / prob
    w0 = (1.0 - tind) / (1.0 - prob)
    if weighted:
        w1 /= w1.mean()
        w0 /= w0.mean()
    return np.column_stack((endog * w0 - params[0], endog * w1 - params[1]))