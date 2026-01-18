import numpy as np
import platform
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_pinned(self):
    machine = platform.machine()
    if machine.startswith('arm') or machine.startswith('aarch64'):
        count = 262144
    else:
        count = 2097152
    A = np.arange(count)
    with cuda.pinned(A):
        self._run_copies(A)