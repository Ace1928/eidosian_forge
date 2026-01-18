import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_return_bool_optional_or_none(self):
    pyfunc = return_bool_optional_or_none
    cfunc = njit((types.int32, types.int32))(pyfunc)
    for x, y in itertools.product((0, 1, 2), (0, 1)):
        self.assertPreciseEqual(pyfunc(x, y), cfunc(x, y))