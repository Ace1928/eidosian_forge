from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps09_fp(x, n):
    return 1 + (1 - n) ** 4 + 4 * n * (1 - n * x) ** 3