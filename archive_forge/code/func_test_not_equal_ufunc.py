import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def test_not_equal_ufunc(self):
    self.signed_unsigned_cmp_test(np.not_equal)