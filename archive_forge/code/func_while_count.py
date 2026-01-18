import math
import numpy as np
from numba import jit
def while_count(s, e):
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c