from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps06_fpp(x, n):
    return -2 * n * n * np.exp(-n * x)