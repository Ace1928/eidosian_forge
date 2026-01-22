from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def _rescale_layout(pos, scale=1):
    lim = 0
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(pos[:, i].max(), lim)
    for i in range(pos.shape[1]):
        pos[:, i] *= scale / lim
    return pos