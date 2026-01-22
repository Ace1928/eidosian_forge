from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def _add_a_b(tests):
    """Add "a" and "b" keys to each test from the "bracket" value"""
    for d in tests:
        for k, v in zip(['a', 'b'], d.get('bracket', [])):
            d[k] = v