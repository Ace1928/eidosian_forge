import unittest
from itertools import product
from numba import types, njit, typed, errors
from numba.tests.support import TestCase
def test_shorthand_syntax(self):

    @njit
    def foo1():
        ty = types.float32[::1, :]
        return typed.List.empty_list(ty)
    self.assertEqual(foo1()._dtype, types.float32[::1, :])

    @njit
    def foo2():
        ty = types.complex64[:, :, :]
        return typed.List.empty_list(ty)
    self.assertEqual(foo2()._dtype, types.complex64[:, :, :])