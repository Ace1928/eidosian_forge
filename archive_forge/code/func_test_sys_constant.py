import math
import sys
import numpy as np
from numba import njit
import numba.tests.usecases as uc
import unittest
def test_sys_constant(self):

    @njit
    def f():
        return sys.hexversion
    self.assertEqual(f(), f.py_func())