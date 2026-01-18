from numba import cuda
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import threading
import unittest
@cuda.jit
def vector_add_scalar(arr, val):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] += val