from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
@cuda.jit(cache=False)
def add_nocache_usecase_kernel(r, x, y):
    r[()] = x[()] + y[()] + Z