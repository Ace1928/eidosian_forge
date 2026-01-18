import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_starargs_non_tuple(self):

    def consumer(*x):
        return x

    @jit(forceobj=True)
    def foo(x):
        return consumer(*x)
    arg = 'ijo'
    got = foo(arg)
    expect = foo.py_func(arg)
    self.assertEqual(got, tuple(arg))
    self.assertEqual(got, expect)