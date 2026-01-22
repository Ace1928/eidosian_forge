from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps14_fp(x, n):
    if x <= 0:
        return 0
    return n / 20.0 * (1.0 / 1.5 + np.cos(x))