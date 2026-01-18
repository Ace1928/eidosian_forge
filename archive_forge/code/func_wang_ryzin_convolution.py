import numpy as np
from scipy.special import erf
def wang_ryzin_convolution(h, Xi, Xj):
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        ordered += wang_ryzin(h, Xi, x) * wang_ryzin(h, Xj, x)
    return ordered