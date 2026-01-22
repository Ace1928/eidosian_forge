import math
import numpy as np
from numba import jit
def andor(x, y):
    return x > 0 and x < 10 or (y > 0 and y < 10)