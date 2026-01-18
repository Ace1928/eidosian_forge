import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_zero_address(self):
    sig = int64()

    @cfunc(sig)
    def test():
        return 123

    class Good(types.WrapperAddressProtocol):
        """A first-class function type with valid address.
            """

        def __wrapper_address__(self):
            return test.address

        def signature(self):
            return sig

    class Bad(types.WrapperAddressProtocol):
        """A first-class function type with invalid 0 address.
            """

        def __wrapper_address__(self):
            return 0

        def signature(self):
            return sig

    class BadToGood(types.WrapperAddressProtocol):
        """A first-class function type with invalid address that is
            recovered to a valid address.
            """
        counter = -1

        def __wrapper_address__(self):
            self.counter += 1
            return test.address * min(1, self.counter)

        def signature(self):
            return sig
    good = Good()
    bad = Bad()
    bad2good = BadToGood()

    @jit(int64(sig.as_type()))
    def foo(func):
        return func()

    @jit(int64())
    def foo_good():
        return good()

    @jit(int64())
    def foo_bad():
        return bad()

    @jit(int64())
    def foo_bad2good():
        return bad2good()
    self.assertEqual(foo(good), 123)
    self.assertEqual(foo_good(), 123)
    with self.assertRaises(ValueError) as cm:
        foo(bad)
    self.assertRegex(str(cm.exception), 'wrapper address of <.*> instance must be a positive')
    with self.assertRaises(RuntimeError) as cm:
        foo_bad()
    self.assertRegex(str(cm.exception), '.* function address is null')
    self.assertEqual(foo_bad2good(), 123)