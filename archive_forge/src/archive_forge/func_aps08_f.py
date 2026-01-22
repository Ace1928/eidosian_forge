from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps08_f(x, n):
    """Degree n polynomial"""
    return x * x - (1 - x) ** n