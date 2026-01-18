import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def test_vectorize_one_signature(self):
    with captured_stdout():
        from numba import vectorize, float64

        @vectorize([float64(float64, float64)])
        def f(x, y):
            return x + y