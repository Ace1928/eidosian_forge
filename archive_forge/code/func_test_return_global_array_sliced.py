import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_return_global_array_sliced(self):
    y = np.ones(4, dtype=np.float32)

    def return_external_array():
        return y[2:]
    cfunc = nrtjit(return_external_array)
    out = cfunc()
    self.assertIsNone(out.base)
    yy = y[2:]
    np.testing.assert_equal(yy, out)
    np.testing.assert_equal(yy, np.ones(2, dtype=np.float32))
    np.testing.assert_equal(out, np.ones(2, dtype=np.float32))