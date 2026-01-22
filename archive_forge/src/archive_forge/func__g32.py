from functools import partial
import numpy as np
def _g32(x):
    ret = np.zeros(len(x))
    m = x < 0.25
    ret[m] = 4 * x[m]
    m = (x >= 0.25) & (x < 0.92)
    ret[m] = -2 * x[m] + 1.84
    m = x >= 0.92
    ret[m] = x[m] / 0.08 - 11.5
    return ret