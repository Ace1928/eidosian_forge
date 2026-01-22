import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
class TestBadEMMPluginVersion(CUDATestCase):
    """
    Ensure that Numba rejects EMM Plugins with incompatible version
    numbers.
    """

    def test_bad_plugin_version(self):
        with self.assertRaises(RuntimeError) as raises:
            cuda.set_memory_manager(BadVersionEMMPlugin)
        self.assertIn('version 1 required', str(raises.exception))