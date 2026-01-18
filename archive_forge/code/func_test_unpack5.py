from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_unpack5(self):
    from numba.tests.chained_assign_usecases import unpack5
    args = [[np.array([2])], [np.array([2.0])]]
    self._test_template(unpack5, args)