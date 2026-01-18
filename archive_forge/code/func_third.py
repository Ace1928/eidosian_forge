from numba import jit
@jit
def third(x):
    return fourth(x) * 4