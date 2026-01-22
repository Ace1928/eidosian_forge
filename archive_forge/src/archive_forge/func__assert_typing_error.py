import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def _assert_typing_error(self, cfunc):
    self.disable_leak_check()
    with self.assertTypingError() as e:
        cfunc(2.2, self.listimpl([3, 2, 1]))
    msg = "First argument 'n' must be an integer"
    self.assertIn(msg, str(e.exception))
    with self.assertTypingError() as e:
        cfunc(2, 100)
    msg = "Second argument 'iterable' must be iterable"
    self.assertIn(msg, str(e.exception))