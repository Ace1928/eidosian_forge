import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_string_comparisons(self):
    import operator
    pyfunc = usecases.string_comparison
    cfunc = jit((types.pyobject, types.pyobject, types.pyobject), forceobj=True)(pyfunc)
    test_str1 = '123'
    test_str2 = '123'
    op = operator.eq
    self.assertEqual(pyfunc(test_str1, test_str2, op), cfunc(test_str1, test_str2, op))
    test_str1 = '123'
    test_str2 = '456'
    op = operator.eq
    self.assertEqual(pyfunc(test_str1, test_str2, op), cfunc(test_str1, test_str2, op))
    test_str1 = '123'
    test_str2 = '123'
    op = operator.ne
    self.assertEqual(pyfunc(test_str1, test_str2, op), cfunc(test_str1, test_str2, op))
    test_str1 = '123'
    test_str2 = '456'
    op = operator.ne
    self.assertEqual(pyfunc(test_str1, test_str2, op), cfunc(test_str1, test_str2, op))