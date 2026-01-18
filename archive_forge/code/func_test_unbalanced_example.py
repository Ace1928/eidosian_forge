import unittest
from numba.tests.support import captured_stdout, skip_parfors_unsupported
from numba import set_parallel_chunksize
from numba.tests.support import TestCase
def test_unbalanced_example(self):
    with captured_stdout():
        from numba import njit, prange
        import numpy as np

        @njit(parallel=True)
        def func1():
            n = 100
            vals = np.empty(n)
            for i in prange(n):
                cur = i + 1
                for j in range(i):
                    if cur % 2 == 0:
                        cur //= 2
                    else:
                        cur = cur * 3 + 1
                vals[i] = cur
            return vals
        result = func1()
        self.assertPreciseEqual(result, func1.py_func())