import time
import ctypes
import numpy as np
from numba.tests.support import captured_stdout
from numba import vectorize, guvectorize
import unittest
class TestParUfuncIssues(unittest.TestCase):
    _numba_parallel_test_ = False

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

    def test_gil_reacquire_deadlock(self):
        """
        Testing issue #1998 due to GIL reacquiring
        """
        proto = ctypes.CFUNCTYPE(None, ctypes.c_int32)
        characters = 'abcdefghij'

        def bar(x):
            print(characters[x])
        cbar = proto(bar)

        @vectorize(['int32(int32)'], target='parallel', nopython=True)
        def foo(x):
            print(x % 10)
            cbar(x % 10)
            return x * 2
        for nelem in [1, 10, 100, 1000]:
            a = np.arange(nelem, dtype=np.int32)
            acopy = a.copy()
            with captured_stdout() as buf:
                got = foo(a)
            stdout = buf.getvalue()
            buf.close()
            got_output = sorted(map(lambda x: x.strip(), stdout.splitlines()))
            expected_output = [str(x % 10) for x in range(nelem)]
            expected_output += [characters[x % 10] for x in range(nelem)]
            expected_output = sorted(expected_output)
            self.assertEqual(got_output, expected_output)
            np.testing.assert_equal(got, 2 * acopy)