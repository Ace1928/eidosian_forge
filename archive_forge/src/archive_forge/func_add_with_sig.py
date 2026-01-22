import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
@jit((types.int32, types.int32))
def add_with_sig(a, b):
    return a + b