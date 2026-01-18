import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_nsmallest_exceptions(self):
    pyfunc = nsmallest
    cfunc = jit(nopython=True)(pyfunc)
    self._assert_typing_error(cfunc)