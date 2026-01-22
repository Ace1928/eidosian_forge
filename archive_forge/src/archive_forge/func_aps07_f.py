from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps07_f(x, n):
    """Upside down parabola with parametrizable height"""
    return (1 + (1 - n) ** 2) * x - (1 - n * x) ** 2