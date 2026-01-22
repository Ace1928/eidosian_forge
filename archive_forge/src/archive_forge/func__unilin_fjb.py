import numpy as np
from scipy.odr._odrpack import Model
def _unilin_fjb(B, x):
    _ret = np.concatenate((x, np.ones(x.shape, float)))
    _ret.shape = (2,) + x.shape
    return _ret