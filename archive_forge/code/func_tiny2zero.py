import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def tiny2zero(x, eps=1e-15):
    """replace abs values smaller than eps by zero, makes copy
    """
    mask = np.abs(x.copy()) < eps
    x[mask] = 0
    return x