from numba import jit
import numpy as np
def unpack5(x):
    a = b, c = inc2(x)
    d, e = f = inc2(x)
    return (a[0] + b / 2 + d + f[0], a[1] + c + e / 2 + f[1])