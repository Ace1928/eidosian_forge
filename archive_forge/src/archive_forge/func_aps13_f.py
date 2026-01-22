from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps13_f(x):
    """Function with *all* derivatives 0 at the root"""
    if x == 0:
        return 0
    y = 1 / x ** 2
    if y > _MAX_EXPABLE:
        return 0
    return x / np.exp(y)