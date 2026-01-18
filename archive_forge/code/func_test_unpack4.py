from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_unpack4(self):
    from numba.tests.chained_assign_usecases import unpack4
    args = [[np.array([1])], [np.array([1.0])]]
    self._test_template(unpack4, args)