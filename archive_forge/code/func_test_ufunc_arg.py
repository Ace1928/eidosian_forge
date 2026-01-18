import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_ufunc_arg(self):

    @vectorize(['f8(f8, f8)'], target='cuda')
    def vadd(a, b):
        return a + b
    h_arr = np.random.random(10)
    arr = ForeignArray(cuda.to_device(h_arr))
    val = 6
    out = vadd(arr, val)
    np.testing.assert_array_equal(out.copy_to_host(), h_arr + val)
    out = ForeignArray(cuda.device_array(h_arr.shape))
    returned = vadd(h_arr, val, out=out)
    np.testing.assert_array_equal(returned.copy_to_host(), h_arr + val)