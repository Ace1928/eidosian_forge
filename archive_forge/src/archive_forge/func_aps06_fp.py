from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps06_fp(x, n):
    return 2 * np.exp(-n) + 2 * n * np.exp(-n * x)