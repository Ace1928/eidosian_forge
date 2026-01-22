import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
class TestNdIter(MemoryLeakMixin, TestCase):
    """
    Test np.nditer()
    """

    def inputs(self):
        yield np.float32(100)
        yield np.array(102, dtype=np.int16)
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order='F')
        a = np.arange(24).reshape((6, 4))[::2]
        yield a

    def basic_inputs(self):
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order='F')

    def check_result(self, got, expected):
        self.assertEqual(set(got), set(expected), (got, expected))

    def test_nditer1(self):
        pyfunc = np_nditer1
        cfunc = jit(nopython=True)(pyfunc)
        for a in self.inputs():
            expected = pyfunc(a)
            got = cfunc(a)
            self.check_result(got, expected)

    def test_nditer2(self):
        pyfunc = np_nditer2
        cfunc = jit(nopython=True)(pyfunc)
        for a, b in itertools.product(self.inputs(), self.inputs()):
            expected = pyfunc(a, b)
            got = cfunc(a, b)
            self.check_result(got, expected)

    def test_nditer3(self):
        pyfunc = np_nditer3
        cfunc = jit(nopython=True)(pyfunc)
        inputs = self.basic_inputs
        for a, b, c in itertools.product(inputs(), inputs(), inputs()):
            expected = pyfunc(a, b, c)
            got = cfunc(a, b, c)
            self.check_result(got, expected)

    def test_errors(self):
        pyfunc = np_nditer2
        cfunc = jit(nopython=True)(pyfunc)
        self.disable_leak_check()

        def check_incompatible(a, b):
            with self.assertRaises(ValueError) as raises:
                cfunc(a, b)
            self.assertIn('operands could not be broadcast together', str(raises.exception))
        check_incompatible(np.arange(2), np.arange(3))
        a = np.arange(12).reshape((3, 4))
        b = np.arange(3)
        check_incompatible(a, b)