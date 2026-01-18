import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def useful_syncwarp(ary):
    i = cuda.grid(1)
    if i == 0:
        ary[0] = 42
    cuda.syncwarp(4294967295)
    ary[i] = ary[0]