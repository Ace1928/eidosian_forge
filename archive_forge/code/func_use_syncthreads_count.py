import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def use_syncthreads_count(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_count(ary_in[i])