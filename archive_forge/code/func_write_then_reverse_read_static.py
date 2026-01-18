from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
@cuda.jit(void(int32[::1], int32[::1]))
def write_then_reverse_read_static(outx, outy):
    arr = cuda.shared.array(nthreads, dtype=test_struct_model_type)
    i = cuda.grid(1)
    ri = nthreads - i - 1
    if i < len(outx) and i < len(outy):
        obj = TestStruct(int32(i), int32(i * 2))
        arr[i] = obj
        cuda.syncthreads()
        outx[i] = arr[ri].x
        outy[i] = arr[ri].y