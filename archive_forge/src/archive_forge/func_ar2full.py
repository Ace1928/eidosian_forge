import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def ar2full(ar):
    """make reduced lagpolynomial into a right side lagpoly array
    """
    nlags, nvar, nvarex = ar.shape
    return np.r_[np.eye(nvar, nvarex)[None, :, :], -ar]