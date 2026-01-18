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
def test_try_to_link_nonexistent(self):
    with self.assertRaises(LinkerError) as e:

        @cuda.jit('void(int32[::1])', link=['nonexistent.a'])
        def f(x):
            x[0] = 0
    self.assertIn('nonexistent.a not found', e.exception.args)