from numba import jit
import numpy as np
def unpack3(x):
    a, b = inc2(x)
    return a + b / 2