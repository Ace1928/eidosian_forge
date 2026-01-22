from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def _isproperdist(X):
    """
    Checks to see if `X` is a proper probability distribution
    """
    X = np.asarray(X)
    if not np.allclose(np.sum(X), 1) or not np.all(X >= 0) or (not np.all(X <= 1)):
        return False
    else:
        return True