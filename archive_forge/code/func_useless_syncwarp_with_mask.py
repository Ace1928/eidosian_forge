import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def useless_syncwarp_with_mask(ary):
    i = cuda.grid(1)
    cuda.syncwarp(65535)
    ary[i] = i