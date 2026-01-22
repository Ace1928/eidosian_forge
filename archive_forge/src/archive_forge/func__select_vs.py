from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _select_vs(v, p):
    """returns the points to use for interpolating v"""
    if v >= 120.0:
        return (60, 120, inf)
    elif v >= 60.0:
        return (40, 60, 120)
    elif v >= 40.0:
        return (30, 40, 60)
    elif v >= 30.0:
        return (24, 30, 40)
    elif v >= 24.0:
        return (20, 24, 30)
    elif v >= 19.5:
        return (19, 20, 24)
    if p >= 0.9:
        if v < 2.5:
            return (1, 2, 3)
    elif v < 3.5:
        return (2, 3, 4)
    vi = int(round(v))
    return (vi - 1, vi, vi + 1)