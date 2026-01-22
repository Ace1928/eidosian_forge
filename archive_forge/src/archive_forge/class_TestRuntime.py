import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
from unittest.mock import patch
class TestRuntime(unittest.TestCase):

    def test_is_supported_version_true(self):
        for v in SUPPORTED_VERSIONS:
            with patch.object(runtime, 'get_version', return_value=v):
                self.assertTrue(runtime.is_supported_version())

    @skip_on_cudasim('The simulator always simulates a supported runtime')
    def test_is_supported_version_false(self):
        for v in ((10, 2), (11, 8), (12, 0)):
            with patch.object(runtime, 'get_version', return_value=v):
                self.assertFalse(runtime.is_supported_version())

    def test_supported_versions(self):
        self.assertEqual(SUPPORTED_VERSIONS, runtime.supported_versions)