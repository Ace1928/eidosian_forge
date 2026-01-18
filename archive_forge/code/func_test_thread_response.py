import time
import ctypes
import numpy as np
from numba.tests.support import captured_stdout
from numba import vectorize, guvectorize
import unittest
def test_thread_response(self):
    """
        Related to #89.
        This does not test #89 but tests the fix for it.
        We want to make sure the worker threads can be used multiple times
        and with different time gap between each execution.
        """

    @vectorize('float64(float64, float64)', target='parallel')
    def fnv(a, b):
        return a + b
    sleep_time = 1
    while sleep_time > 1e-05:
        time.sleep(sleep_time)
        a = b = np.arange(10 ** 5)
        np.testing.assert_equal(a + b, fnv(a, b))
        sleep_time /= 2