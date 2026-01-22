import numpy as np
from scipy.odr._odrpack import Model
def _exp_fjd(B, x):
    return B[1] * np.exp(B[1] * x)