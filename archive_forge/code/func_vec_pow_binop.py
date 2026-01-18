import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def vec_pow_binop(r, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = x[i] ** y[i]