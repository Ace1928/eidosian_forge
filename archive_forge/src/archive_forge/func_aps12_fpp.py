from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps12_fpp(x, n):
    return np.power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n