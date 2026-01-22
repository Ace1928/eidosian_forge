from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps15_f(x, n):
    """piecewise linear, constant outside of [0, 0.002/(1+n)]"""
    if x < 0:
        return -0.859
    if x > 2 * 0.001 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) - 1.859