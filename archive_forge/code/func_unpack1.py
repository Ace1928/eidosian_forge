from numba import jit
import numpy as np
def unpack1(x, y):
    a, b = (x, y)
    return a + b / 2