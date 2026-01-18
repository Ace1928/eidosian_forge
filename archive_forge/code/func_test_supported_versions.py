import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
from unittest.mock import patch
def test_supported_versions(self):
    self.assertEqual(SUPPORTED_VERSIONS, runtime.supported_versions)