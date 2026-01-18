import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_zero_size_array(self):
    c_arr = cuda.device_array(0)
    self.assertEqual(c_arr.__cuda_array_interface__['data'][0], 0)

    @cuda.jit
    def add_one(arr):
        x = cuda.grid(1)
        N = arr.shape[0]
        if x < N:
            arr[x] += 1
    d_arr = ForeignArray(c_arr)
    add_one[1, 10](d_arr)