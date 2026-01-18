import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
@cuda.jit
def use_activemask(x):
    i = cuda.grid(1)
    if i % 2 == 0:
        x[i] = cuda.activemask()
    else:
        x[i] = cuda.activemask()