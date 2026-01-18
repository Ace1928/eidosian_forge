import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_npy_sqrt(self):
    pyfunc = npy_sqrt
    x_values = [2, 1, 2, 2, 1, 2, 0.1, 0.2]
    x_types = [types.int16, types.uint16]
    self.run_unary(pyfunc, x_types, x_values, prec='single')
    x_types = [types.int32, types.int64, types.uint32, types.uint64, types.float32, types.float64]
    self.run_unary(pyfunc, x_types, x_values)