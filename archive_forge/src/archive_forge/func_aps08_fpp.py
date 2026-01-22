from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps08_fpp(x, n):
    return 2 - n * (n - 1) * (1 - x) ** (n - 2)