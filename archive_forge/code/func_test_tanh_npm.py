import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_tanh_npm(self):
    self.check_unary_func(tanh_usecase, enable_pyobj_flags, ulps=2)