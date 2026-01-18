import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def test_rad2deg_ufunc(self):
    self.basic_ufunc_test(np.rad2deg, kinds='f')