import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
@jit(nopython=True)
def add_nopython_fail(a, b):
    object()
    return a + b