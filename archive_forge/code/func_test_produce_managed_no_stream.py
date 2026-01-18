import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
@linux_only
def test_produce_managed_no_stream(self):
    managed_arr = cuda.managed_array(10)
    self.assertIsNone(managed_arr.__cuda_array_interface__['stream'])