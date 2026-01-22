import numpy as np
from scipy.odr._odrpack import Model
def _quad_fjb(B, x):
    _ret = np.concatenate((x * x, x, np.ones(x.shape, float)))
    _ret.shape = (3,) + x.shape
    return _ret