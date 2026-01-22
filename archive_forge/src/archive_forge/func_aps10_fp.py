from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps10_fp(x, n):
    return np.exp(-n * x) * (-n * (x - 1) + 1) + n * x ** (n - 1)