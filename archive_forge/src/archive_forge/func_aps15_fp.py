from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps15_fp(x, n):
    if not 0 <= x <= 2 * 0.001 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000