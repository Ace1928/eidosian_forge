import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def _make_ufunc_usecase(self, ufunc):
    return _make_ufunc_usecase(ufunc)