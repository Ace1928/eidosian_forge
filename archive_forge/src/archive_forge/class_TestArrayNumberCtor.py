import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
class TestArrayNumberCtor(TestCase):
    """
    Test <number class>(some sequence)
    """

    def check_type_constructor(self, np_type, values):
        pyfunc = converter(np_type)
        cfunc = jit(nopython=True)(pyfunc)
        for val in values:
            expected = np_type(val)
            got = cfunc(val)
            self.assertPreciseEqual(got, expected)

    def test_1d(self):
        values = [(1.0, 2.5), (1, 2.5), [1.0, 2.5], ()]
        for tp_name in real_np_types():
            np_type = getattr(np, tp_name)
            self.check_type_constructor(np_type, values)
        values = [(1j, 2.5), [1.0, 2.5]]
        for tp_name in complex_np_types():
            np_type = getattr(np, tp_name)
            self.check_type_constructor(np_type, values)

    def test_2d(self):
        values = [((1.0, 2.5), (3.5, 4)), [(1.0, 2.5), (3.5, 4.0)], ([1.0, 2.5], [3.5, 4.0]), [(), ()]]
        for tp_name in real_np_types():
            np_type = getattr(np, tp_name)
            self.check_type_constructor(np_type, values)
        for tp_name in complex_np_types():
            np_type = getattr(np, tp_name)
            self.check_type_constructor(np_type, values)