from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _isfloat(x):
    """
    returns True if x is a float,
    returns False otherwise
    """
    try:
        float(x)
    except:
        return False
    return True