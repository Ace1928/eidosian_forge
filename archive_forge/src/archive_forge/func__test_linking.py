import numpy as np
import warnings
from numba.cuda.testing import unittest
from numba.cuda.testing import (skip_on_cudasim, skip_if_cuda_includes_missing)
from numba.cuda.testing import CUDATestCase, test_data_dir
from numba.cuda.cudadrv.driver import (CudaAPIError, Linker,
from numba.cuda.cudadrv.error import NvrtcError
from numba.cuda import require_context
from numba.tests.support import ignore_internal_warnings
from numba import cuda, void, float64, int64, int32, typeof, float32
def _test_linking(self, eager):
    global bar
    bar = cuda.declare_device('bar', 'int32(int32)')
    link = str(test_data_dir / 'jitlink.ptx')
    if eager:
        args = ['void(int32[:], int32[:])']
    else:
        args = []

    @cuda.jit(*args, link=[link])
    def foo(x, y):
        i = cuda.grid(1)
        x[i] += bar(y[i])
    A = np.array([123], dtype=np.int32)
    B = np.array([321], dtype=np.int32)
    foo[1, 1](A, B)
    self.assertTrue(A[0] == 123 + 2 * 321)