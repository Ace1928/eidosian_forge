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
def test_return_global_array(self):
    y = np.ones(4, dtype=np.float32)
    initrefct = sys.getrefcount(y)

    def return_external_array():
        return y
    cfunc = nrtjit(return_external_array)
    out = cfunc()
    self.assertEqual(initrefct + 1, sys.getrefcount(y))
    np.testing.assert_equal(y, out)
    np.testing.assert_equal(y, np.ones(4, dtype=np.float32))
    np.testing.assert_equal(out, np.ones(4, dtype=np.float32))
    del out
    gc.collect()
    self.assertEqual(initrefct + 1, sys.getrefcount(y))
    del cfunc
    gc.collect()
    self.assertEqual(initrefct, sys.getrefcount(y))