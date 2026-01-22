from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps12_f(x, n):
    """nth root of x, with a zero at x=n"""
    return np.power(x, 1.0 / n) - np.power(n, 1.0 / n)