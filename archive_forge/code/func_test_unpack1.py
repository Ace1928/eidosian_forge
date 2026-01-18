from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_unpack1(self):
    from numba.tests.chained_assign_usecases import unpack1
    args = [[1, 3.0], [1.0, 3]]
    self._test_template(unpack1, args)