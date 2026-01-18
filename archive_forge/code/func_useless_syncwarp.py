import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def useless_syncwarp(ary):
    i = cuda.grid(1)
    cuda.syncwarp()
    ary[i] = i