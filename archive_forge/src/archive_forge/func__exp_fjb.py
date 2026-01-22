import numpy as np
from scipy.odr._odrpack import Model
def _exp_fjb(B, x):
    res = np.concatenate((np.ones(x.shape[-1], float), x * np.exp(B[1] * x)))
    res.shape = (2, x.shape[-1])
    return res