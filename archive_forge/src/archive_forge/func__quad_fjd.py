import numpy as np
from scipy.odr._odrpack import Model
def _quad_fjd(B, x):
    return 2 * x * B[0] + B[1]