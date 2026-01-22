from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _psturng_scalar(q, r, v):
    return np.squeeze(_psturng(q, r, v))