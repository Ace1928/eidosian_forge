import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
class ArrayTest(parameterized.TestCase):

    @parameterized.product(scalar_type=INT4_TYPES)
    def testDtype(self, scalar_type):
        self.assertEqual(scalar_type, np.dtype(scalar_type))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testDeepCopyDoesNotAlterHash(self, scalar_type):
        dtype = np.dtype(scalar_type)
        h = hash(dtype)
        _ = copy.deepcopy(dtype)
        self.assertEqual(h, hash(dtype))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testArray(self, scalar_type):
        x = np.array([[1, 2, 3]], dtype=scalar_type)
        self.assertEqual(scalar_type, x.dtype)
        self.assertEqual('[[1 2 3]]', str(x))
        np.testing.assert_array_equal(x, x)
        self.assertTrue((x == x).all())

    @parameterized.product(scalar_type=INT4_TYPES, ufunc=[np.nonzero, np.logical_not])
    def testUnaryPredicateUfunc(self, scalar_type, ufunc):
        x = np.array(VALUES[scalar_type])
        y = np.array(VALUES[scalar_type], dtype=scalar_type)
        np.testing.assert_array_equal(ufunc(x), ufunc(y))

    @parameterized.product(scalar_type=INT4_TYPES, ufunc=[np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal, np.logical_and, np.logical_or, np.logical_xor])
    def testPredicateUfuncs(self, scalar_type, ufunc):
        x = np.array(VALUES[scalar_type])
        y = np.array(VALUES[scalar_type], dtype=scalar_type)
        np.testing.assert_array_equal(ufunc(x[:, None], x[None, :]), ufunc(y[:, None], y[None, :]))

    @parameterized.product(scalar_type=INT4_TYPES, dtype=[np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16, np.int32, np.int64, np.complex64, np.complex128, np.clongdouble, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_, np.longlong, np.uintc, np.ulonglong])
    def testCasts(self, scalar_type, dtype):
        x_orig = np.array(VALUES[scalar_type])
        x = np.array(VALUES[scalar_type]).astype(dtype)
        x = np.where(x == x_orig, x, np.zeros_like(x))
        y = x.astype(scalar_type)
        z = y.astype(dtype)
        self.assertTrue(np.all(x == y), msg=(x, y))
        self.assertEqual(scalar_type, y.dtype)
        self.assertTrue(np.all(x == z))
        self.assertEqual(dtype, z.dtype)

    @parameterized.product(scalar_type=INT4_TYPES, ufunc=[np.add, np.subtract, np.multiply, np.floor_divide, np.remainder])
    @ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
    def testBinaryUfuncs(self, scalar_type, ufunc):
        x = np.array(VALUES[scalar_type])
        y = np.array(VALUES[scalar_type], dtype=scalar_type)
        np.testing.assert_array_equal(ufunc(x[:, None], x[None, :]).astype(scalar_type), ufunc(y[:, None], y[None, :]))