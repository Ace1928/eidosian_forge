from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_unpack2(self):
    from numba.tests.chained_assign_usecases import unpack2
    args = [[np.array([2]), np.array([4.0])], [np.array([2.0]), np.array([4])]]
    self._test_template(unpack2, args)