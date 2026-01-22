import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.named_parameters(({'testcase_name': '_' + dtype.__name__, 'float_type': dtype} for dtype in FLOAT_DTYPES))
class CustomFloatTest(parameterized.TestCase):
    """Tests the non-numpy Python methods of the custom float type."""

    def testModuleName(self, float_type):
        self.assertEqual(float_type.__module__, 'ml_dtypes')

    def testPickleable(self, float_type):
        x = np.arange(10, dtype=float_type)
        serialized = pickle.dumps(x)
        x_out = pickle.loads(serialized)
        self.assertEqual(x_out.dtype, x.dtype)
        np.testing.assert_array_equal(x_out.astype('float32'), x.astype('float32'))

    def testRoundTripToFloat(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            np.testing.assert_equal(v, float(float_type(v)))

    @ignore_warning(category=RuntimeWarning, message='overflow encountered')
    def testRoundTripNumpyTypes(self, float_type):
        for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
            for f in FLOAT_VALUES[float_type]:
                np.testing.assert_equal(dtype(f), dtype(float_type(dtype(f))))
                np.testing.assert_equal(float(dtype(f)), float(float_type(dtype(f))))
                np.testing.assert_equal(dtype(f), dtype(float_type(np.array(f, dtype))))
            np.testing.assert_equal(dtype(np.array(FLOAT_VALUES[float_type], float_type)), np.array(FLOAT_VALUES[float_type], dtype))

    def testRoundTripToInt(self, float_type):
        for v in INT_VALUES[float_type]:
            self.assertEqual(v, int(float_type(v)))
            self.assertEqual(-v, int(float_type(-v)))

    @ignore_warning(category=RuntimeWarning, message='overflow encountered')
    def testRoundTripToNumpy(self, float_type):
        for dtype in [float_type, np.float16, np.float32, np.float64, np.longdouble]:
            with self.subTest(dtype.__name__):
                for v in FLOAT_VALUES[float_type]:
                    np.testing.assert_equal(dtype(v), dtype(float_type(dtype(v))))
                    np.testing.assert_equal(dtype(v), dtype(float_type(dtype(v))))
                    np.testing.assert_equal(dtype(v), dtype(float_type(np.array(v, dtype))))
                if dtype != float_type:
                    np.testing.assert_equal(np.array(FLOAT_VALUES[float_type], dtype), float_type(np.array(FLOAT_VALUES[float_type], dtype)).astype(dtype))

    def testBetweenCustomTypes(self, float_type):
        for dtype in FLOAT_DTYPES:
            x = np.array(FLOAT_VALUES[float_type], dtype=dtype)
            y = x.astype(float_type)
            z = x.astype(float).astype(float_type)
            numpy_assert_allclose(y, z, float_type=float_type)

    def testStr(self, float_type):
        for value in FLOAT_VALUES[float_type]:
            self.assertEqual('%.6g' % float(float_type(value)), str(float_type(value)))

    def testFromStr(self, float_type):
        self.assertEqual(float_type(1.2), float_type('1.2'))
        self.assertTrue(np.isnan(float_type('nan')))
        self.assertTrue(np.isnan(float_type('-nan')))
        if dtype_has_inf(float_type):
            self.assertEqual(float_type(float('inf')), float_type('inf'))
            self.assertEqual(float_type(float('-inf')), float_type('-inf'))

    def testRepr(self, float_type):
        for value in FLOAT_VALUES[float_type]:
            self.assertEqual('%.6g' % float(float_type(value)), repr(float_type(value)))

    def testItem(self, float_type):
        self.assertIsInstance(float_type(0).item(), float)

    def testHashZero(self, float_type):
        """Tests that negative zero and zero hash to the same value."""
        self.assertEqual(hash(float_type(-0.0)), hash(float_type(0.0)))

    def testHashNumbers(self, float_type):
        for value in np.extract(np.isfinite(FLOAT_VALUES[float_type]), FLOAT_VALUES[float_type]):
            with self.subTest(value):
                self.assertEqual(hash(value), hash(float_type(value)), str(value))

    def testHashNan(self, float_type):
        for name, nan in [('PositiveNan', float_type(float('nan'))), ('NegativeNan', float_type(float('-nan')))]:
            with self.subTest(name):
                nan_hash = hash(nan)
                nan_object_hash = object.__hash__(nan)
                self.assertIn(nan_hash, (sys.hash_info.nan, nan_object_hash), str(nan))

    def testHashInf(self, float_type):
        if dtype_has_inf(float_type):
            self.assertEqual(sys.hash_info.inf, hash(float_type(float('inf'))), 'inf')
            self.assertEqual(-sys.hash_info.inf, hash(float_type(float('-inf'))), '-inf')

    def testNegate(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            np.testing.assert_equal(float(float_type(-float(float_type(v)))), float(-float_type(v)))

    def testAdd(self, float_type):
        for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25), (float('inf'), -2.25), (float('-inf'), -2.25), (3.5, float('nan'))]:
            binary_operation_test(a, b, op=lambda a, b: a + b, float_type=float_type)

    def testAddScalarTypePromotion(self, float_type):
        """Tests type promotion against Numpy scalar values."""
        types = [float_type, np.float16, np.float32, np.float64, np.longdouble]
        for lhs_type in types:
            for rhs_type in types:
                expected_type = numpy_promote_types(lhs_type, rhs_type, float_type=float_type, next_largest_fp_type=np.float32)
                actual_type = type(lhs_type(3.5) + rhs_type(2.25))
                self.assertEqual(expected_type, actual_type)

    def testAddArrayTypePromotion(self, float_type):
        self.assertEqual(np.float32, type(float_type(3.5) + np.array(2.25, np.float32)))
        self.assertEqual(np.float32, type(np.array(3.5, np.float32) + float_type(2.25)))

    def testSub(self, float_type):
        for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25), (-2.25, float('inf')), (-2.25, float('-inf')), (3.5, float('nan'))]:
            binary_operation_test(a, b, op=lambda a, b: a - b, float_type=float_type)

    def testMul(self, float_type):
        for a, b in [(0, 0), (1, 0), (1, -1), (3.5, -2.25), (float('inf'), -2.25), (float('-inf'), -2.25), (3.5, float('nan'))]:
            binary_operation_test(a, b, op=lambda a, b: a * b, float_type=float_type)

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered')
    @ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
    def testDiv(self, float_type):
        for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25), (float('inf'), -2.25), (float('-inf'), -2.25), (3.5, float('nan'))]:
            binary_operation_test(a, b, op=lambda a, b: a / b, float_type=float_type)

    def testLess(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v < w, float_type(v) < float_type(w))

    def testLessEqual(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v <= w, float_type(v) <= float_type(w))

    def testGreater(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v > w, float_type(v) > float_type(w))

    def testGreaterEqual(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v >= w, float_type(v) >= float_type(w))

    def testEqual(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v == w, float_type(v) == float_type(w))

    def testNotEqual(self, float_type):
        for v in FLOAT_VALUES[float_type]:
            for w in FLOAT_VALUES[float_type]:
                self.assertEqual(v != w, float_type(v) != float_type(w))

    def testNan(self, float_type):
        a = np.isnan(float_type(float('nan')))
        self.assertTrue(a)
        numpy_assert_allclose(np.array([1.0, a]), np.array([1.0, a]), float_type=float_type)
        a = np.array([float_type(1.34375), float_type(1.4375), float_type(float('nan'))], dtype=float_type)
        b = np.array([float_type(1.3359375), float_type(1.4375), float_type(float('nan'))], dtype=float_type)
        numpy_assert_allclose(a, b, rtol=0.1, atol=0.1, equal_nan=True, err_msg='', verbose=True, float_type=float_type)

    def testSort(self, float_type):
        values_to_sort = np.float32([x for x in FLOAT_VALUES[float_type] if not np.isnan(x)])
        sorted_f32 = np.sort(values_to_sort)
        sorted_float_type = np.sort(values_to_sort.astype(float_type))
        np.testing.assert_equal(sorted_f32, np.float32(sorted_float_type))

    def testArgmax(self, float_type):
        values_to_sort = np.float32(float_type(np.float32(FLOAT_VALUES[float_type])))
        argmax_f32 = np.argmax(values_to_sort)
        argmax_float_type = np.argmax(values_to_sort.astype(float_type))
        np.testing.assert_equal(argmax_f32, argmax_float_type)

    def testArgmaxOnNan(self, float_type):
        """Ensures we return the right thing for multiple NaNs."""
        one_with_nans = np.array([1.0, float('nan'), float('nan')], dtype=np.float32)
        np.testing.assert_equal(np.argmax(one_with_nans.astype(float_type)), np.argmax(one_with_nans))

    def testArgmaxOnNegativeInfinity(self, float_type):
        """Ensures we return the right thing for negative infinities."""
        inf = np.array([float('-inf')], dtype=np.float32)
        np.testing.assert_equal(np.argmax(inf.astype(float_type)), np.argmax(inf))

    def testArgmin(self, float_type):
        values_to_sort = np.float32(float_type(np.float32(FLOAT_VALUES[float_type])))
        argmin_f32 = np.argmin(values_to_sort)
        argmin_float_type = np.argmin(values_to_sort.astype(float_type))
        np.testing.assert_equal(argmin_f32, argmin_float_type)

    def testArgminOnNan(self, float_type):
        """Ensures we return the right thing for multiple NaNs."""
        one_with_nans = np.array([1.0, float('nan'), float('nan')], dtype=np.float32)
        np.testing.assert_equal(np.argmin(one_with_nans.astype(float_type)), np.argmin(one_with_nans))

    def testArgminOnPositiveInfinity(self, float_type):
        """Ensures we return the right thing for positive infinities."""
        inf = np.array([float('inf')], dtype=np.float32)
        np.testing.assert_equal(np.argmin(inf.astype(float_type)), np.argmin(inf))

    def testDtypeFromString(self, float_type):
        assert np.dtype(float_type.__name__) == np.dtype(float_type)