from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps05_f(x):
    """Simple Trigonometric function"""
    return np.sin(x) - 1.0 / 2