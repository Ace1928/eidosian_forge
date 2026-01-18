import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def usecase3(x, N):
    """
    Base on test2 of https://github.com/numba/numba/issues/381
    """
    for k in range(N):
        print(x.f1[k], x.s1[k], x.f2[k])