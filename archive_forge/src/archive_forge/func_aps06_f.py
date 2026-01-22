from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps06_f(x, n):
    """Exponential rapidly changing from -1 to 1 at x=0"""
    return 2 * x * np.exp(-n) - 2 * np.exp(-n * x) + 1