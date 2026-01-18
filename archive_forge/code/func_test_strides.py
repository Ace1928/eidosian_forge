import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_strides(self):
    c_arr = cuda.device_array((2, 3, 4))
    self.assertEqual(c_arr.__cuda_array_interface__['strides'], None)
    c_arr = c_arr[:, 1, :]
    self.assertNotEqual(c_arr.__cuda_array_interface__['strides'], None)