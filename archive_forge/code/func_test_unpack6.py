from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_unpack6(self):
    from numba.tests.chained_assign_usecases import unpack6
    args1 = (3.0, 2)
    args2 = (3.0, 2.0)
    self._test_template(unpack6, [args1, args2])