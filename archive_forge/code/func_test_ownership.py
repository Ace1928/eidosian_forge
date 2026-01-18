import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
@skip_if_external_memmgr('Ownership not relevant with external memmgr')
def test_ownership(self):
    ctx = cuda.current_context()
    deallocs = ctx.memory_manager.deallocations
    deallocs.clear()
    self.assertEqual(len(deallocs), 0)
    d_arr = cuda.to_device(np.arange(100))
    cvted = cuda.as_cuda_array(d_arr)
    del d_arr
    self.assertEqual(len(deallocs), 0)
    np.testing.assert_equal(cvted.copy_to_host(), np.arange(100))
    del cvted
    self.assertEqual(len(deallocs), 1)
    deallocs.clear()