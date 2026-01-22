from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps09_f(x, n):
    """Upside down quartic with parametrizable height"""
    return (1 + (1 - n) ** 4) * x - (1 - n * x) ** 4