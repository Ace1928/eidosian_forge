import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_independent_scheduling(arr):
    i = cuda.threadIdx.x
    if i % 4 == 0:
        ballot = cuda.ballot_sync(286331153, True)
    elif i % 4 == 1:
        ballot = cuda.ballot_sync(572662306, True)
    elif i % 4 == 2:
        ballot = cuda.ballot_sync(1145324612, True)
    elif i % 4 == 3:
        ballot = cuda.ballot_sync(2290649224, True)
    arr[i] = ballot