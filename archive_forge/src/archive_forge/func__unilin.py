import numpy as np
from scipy.odr._odrpack import Model
def _unilin(B, x):
    return x * B[0] + B[1]