import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_use_dict_iterable_args(self):

    @njit
    def dict_iterable_1(a, b):
        d = dict(zip(a, b))
        return d

    @njit
    def dict_iterable_2():
        return dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
    inps = (([1, 2, 3], [4, 5, 6]), (np.arange(4), np.arange(4)), ([1, 2, 3], 'abc'), ([1, 2, 3, 4], 'abc'))
    for a, b in inps:
        d = dict_iterable_1(a, b)
        self.assertEqual(d, dict(zip(a, b)))
    self.assertEqual(dict_iterable_2(), dict_iterable_2.py_func())