import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_match_any_sync(ary_in, ary_out):
    i = cuda.grid(1)
    ballot = cuda.match_any_sync(4294967295, ary_in[i])
    ary_out[i] = ballot