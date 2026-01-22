import numpy as np
from scipy.odr._odrpack import Model
def _unilin_fjd(B, x):
    return np.ones(x.shape, float) * B[0]