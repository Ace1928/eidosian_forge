from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps14_fpp(x, n):
    if x <= 0:
        return 0
    return -n / 20.0 * np.sin(x)