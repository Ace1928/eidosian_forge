import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
class TestIntEnum(BaseEnumTest, TestCase):
    """
    Tests for IntEnum classes and members.
    """
    values = [Shape.circle, Shape.square]
    pairs = [(Shape.circle, Shape.circle), (Shape.circle, Shape.square), (RequestError.not_found, RequestError.not_found), (RequestError.internal_error, RequestError.not_found)]

    def test_int_coerce(self):
        pyfunc = int_coerce_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for arg in [300, 450, 550]:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_int_cast(self):
        pyfunc = int_cast_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for arg in [300, 450, 550]:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_vectorize(self):
        cfunc = vectorize(nopython=True)(vectorize_usecase)
        arg = np.array([2, 404, 500, 404])
        sol = np.array([vectorize_usecase(i) for i in arg], dtype=arg.dtype)
        self.assertPreciseEqual(sol, cfunc(arg))

    def test_hash(self):

        def pyfun(x):
            return hash(x)
        cfunc = jit(nopython=True)(pyfun)
        for member in IntEnumWithNegatives:
            self.assertPreciseEqual(pyfun(member), cfunc(member))

    def test_int_shape_cast(self):

        def pyfun_empty(x):
            return np.empty((x, x), dtype='int64').fill(-1)

        def pyfun_zeros(x):
            return np.zeros((x, x), dtype='int64')

        def pyfun_ones(x):
            return np.ones((x, x), dtype='int64')
        for pyfun in [pyfun_empty, pyfun_zeros, pyfun_ones]:
            cfunc = jit(nopython=True)(pyfun)
            for member in IntEnumWithNegatives:
                if member >= 0:
                    self.assertPreciseEqual(pyfun(member), cfunc(member))