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
class CustomFloatNumPyTest(parameterized.TestCase):
    """Tests NumPy integration of the custom float types."""

    def testDtype(self, float_type):
        self.assertEqual(float_type, np.dtype(float_type))

    def testDeepCopyDoesNotAlterHash(self, float_type):
        dtype = np.dtype(float_type)
        h = hash(dtype)
        _ = copy.deepcopy(dtype)
        self.assertEqual(h, hash(dtype))

    def testArray(self, float_type):
        x = np.array([[1, 2, 3]], dtype=float_type)
        self.assertEqual(float_type, x.dtype)
        self.assertEqual('[[1 2 3]]', str(x))
        np.testing.assert_equal(x, x)
        numpy_assert_allclose(x, x, float_type=float_type)
        self.assertTrue((x == x).all())

    def testComparisons(self, float_type):
        x = np.array([30, 7, -30], dtype=np.float32)
        bx = x.astype(float_type)
        y = np.array([17, 7, 0], dtype=np.float32)
        by = y.astype(float_type)
        np.testing.assert_equal(x == y, bx == by)
        np.testing.assert_equal(x != y, bx != by)
        np.testing.assert_equal(x < y, bx < by)
        np.testing.assert_equal(x > y, bx > by)
        np.testing.assert_equal(x <= y, bx <= by)
        np.testing.assert_equal(x >= y, bx >= by)

    def testEqual2(self, float_type):
        a = np.array([31], float_type)
        b = np.array([15], float_type)
        self.assertFalse(a.__eq__(b))

    def testCanCast(self, float_type):
        allowed_casts = [(np.bool_, float_type), (np.int8, float_type), (np.uint8, float_type), (float_type, np.float32), (float_type, np.float64), (float_type, np.longdouble), (float_type, np.complex64), (float_type, np.complex128), (float_type, np.clongdouble)]
        all_dtypes = [np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16, np.int32, np.int64, np.complex64, np.complex128, np.clongdouble, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_, np.longlong, np.uintc, np.ulonglong]
        for d in all_dtypes:
            with self.subTest(d.__name__):
                self.assertEqual((float_type, d) in allowed_casts, np.can_cast(float_type, d))
                self.assertEqual((d, float_type) in allowed_casts, np.can_cast(d, float_type))

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered in cast')
    def testCasts(self, float_type):
        for dtype in [np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16, np.int32, np.int64, np.complex64, np.complex128, np.clongdouble, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_, np.longlong, np.uintc, np.ulonglong]:
            x = np.array([[1, 2, 3]], dtype=dtype)
            y = x.astype(float_type)
            z = y.astype(dtype)
            self.assertTrue(np.all(x == y))
            self.assertEqual(float_type, y.dtype)
            self.assertTrue(np.all(x == z))
            self.assertEqual(dtype, z.dtype)

    @ignore_warning(category=np.ComplexWarning)
    def testConformNumpyComplex(self, float_type):
        for dtype in [np.complex64, np.complex128, np.clongdouble]:
            x = np.array([1.5, 2.5 + 2j, 3.5], dtype=dtype)
            y_np = x.astype(np.float32)
            y_tf = x.astype(float_type)
            numpy_assert_allclose(y_np, y_tf, atol=0.02, float_type=float_type)
            z_np = y_np.astype(dtype)
            z_tf = y_tf.astype(dtype)
            numpy_assert_allclose(z_np, z_tf, atol=0.02, float_type=float_type)

    def testArange(self, float_type):
        np.testing.assert_equal(np.arange(100, dtype=np.float32).astype(float_type), np.arange(100, dtype=float_type))
        np.testing.assert_equal(np.arange(-8, 8, 1, dtype=np.float32).astype(float_type), np.arange(-8, 8, 1, dtype=float_type))
        np.testing.assert_equal(np.arange(-0.0, -2.0, -0.25, dtype=np.float32).astype(float_type), np.arange(-0.0, -2.0, -0.25, dtype=float_type))
        np.testing.assert_equal(np.arange(-16.0, 16.0, 2.0, dtype=np.float32).astype(float_type), np.arange(-16.0, 16.0, 2.0, dtype=float_type))

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered')
    @ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
    def testUnaryUfunc(self, float_type):
        for op in UNARY_UFUNCS:
            with self.subTest(op.__name__):
                rng = np.random.RandomState(seed=42)
                x = rng.randn(3, 7, 10).astype(float_type)
                numpy_assert_allclose(op(x).astype(np.float32), truncate(op(x.astype(np.float32)), float_type=float_type), rtol=0.0001, float_type=float_type)

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered')
    @ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
    def testBinaryUfunc(self, float_type):
        for op in BINARY_UFUNCS:
            with self.subTest(op.__name__):
                rng = np.random.RandomState(seed=42)
                x = rng.randn(3, 7, 10).astype(float_type)
                y = rng.randn(4, 1, 7, 10).astype(float_type)
                numpy_assert_allclose(op(x, y).astype(np.float32), truncate(op(x.astype(np.float32), y.astype(np.float32)), float_type=float_type), rtol=0.0001, float_type=float_type)

    def testBinaryPredicateUfunc(self, float_type):
        for op in BINARY_PREDICATE_UFUNCS:
            with self.subTest(op.__name__):
                rng = np.random.RandomState(seed=42)
                x = rng.randn(3, 7).astype(float_type)
                y = rng.randn(4, 1, 7).astype(float_type)
                np.testing.assert_equal(op(x, y), op(x.astype(np.float32), y.astype(np.float32)))

    def testPredicateUfunc(self, float_type):
        for op in [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not]:
            with self.subTest(op.__name__):
                rng = np.random.RandomState(seed=42)
                shape = (3, 7, 10)
                posinf_flips = rng.rand(*shape) < 0.1
                neginf_flips = rng.rand(*shape) < 0.1
                nan_flips = rng.rand(*shape) < 0.1
                vals = rng.randn(*shape)
                vals = np.where(posinf_flips, np.inf, vals)
                vals = np.where(neginf_flips, -np.inf, vals)
                vals = np.where(nan_flips, np.nan, vals)
                vals = vals.astype(float_type)
                np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))

    def testDivmod(self, float_type):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7).astype(float_type)
        y = rng.randn(4, 1, 7).astype(float_type)
        o1, o2 = np.divmod(x, y)
        e1, e2 = np.divmod(x.astype(np.float32), y.astype(np.float32))
        numpy_assert_allclose(o1, truncate(e1, float_type=float_type), rtol=0.01, float_type=float_type)
        numpy_assert_allclose(o2, truncate(e2, float_type=float_type), rtol=0.01, float_type=float_type)

    def testModf(self, float_type):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7).astype(float_type)
        o1, o2 = np.modf(x)
        e1, e2 = np.modf(x.astype(np.float32))
        numpy_assert_allclose(o1.astype(np.float32), truncate(e1, float_type=float_type), rtol=0.01, float_type=float_type)
        numpy_assert_allclose(o2.astype(np.float32), truncate(e2, float_type=float_type), rtol=0.01, float_type=float_type)

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered')
    def testLdexp(self, float_type):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7).astype(float_type)
        y = rng.randint(-50, 50, (1, 7)).astype(np.int32)
        self.assertEqual(np.ldexp(x, y).dtype, x.dtype)
        numpy_assert_allclose(np.ldexp(x, y).astype(np.float32), truncate(np.ldexp(x.astype(np.float32), y), float_type=float_type), rtol=0.01, atol=1e-06, float_type=float_type)

    def testFrexp(self, float_type):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7).astype(float_type)
        mant1, exp1 = np.frexp(x)
        mant2, exp2 = np.frexp(x.astype(np.float32))
        np.testing.assert_equal(exp1, exp2)
        numpy_assert_allclose(mant1, mant2, rtol=0.01, float_type=float_type)

    def testCopySign(self, float_type):
        for bits in list(range(1, 128)):
            with self.subTest(bits):
                bits_type = BITS_TYPE[float_type]
                val = bits_type(bits).view(float_type)
                val_with_sign = np.copysign(val, float_type(-1))
                val_with_sign_bits = val_with_sign.view(bits_type)
                num_bits = np.iinfo(bits_type).bits
                np.testing.assert_equal(bits | 1 << num_bits - 1, val_with_sign_bits)

    def testNextAfter(self, float_type):
        one = np.array(1.0, dtype=float_type)
        two = np.array(2.0, dtype=float_type)
        zero = np.array(0.0, dtype=float_type)
        nan = np.array(np.nan, dtype=float_type)
        np.testing.assert_equal(np.nextafter(one, two) - one, ml_dtypes.finfo(float_type).eps)
        np.testing.assert_equal(np.nextafter(one, zero) - one, -ml_dtypes.finfo(float_type).eps / 2)
        np.testing.assert_equal(np.isnan(np.nextafter(nan, one)), True)
        np.testing.assert_equal(np.isnan(np.nextafter(one, nan)), True)
        np.testing.assert_equal(np.nextafter(one, one), one)
        smallest_denormal = ml_dtypes.finfo(float_type).smallest_subnormal
        np.testing.assert_equal(np.nextafter(zero, one), smallest_denormal)
        np.testing.assert_equal(np.nextafter(zero, -one), -smallest_denormal)
        for a, b in itertools.permutations([0.0, nan], 2):
            np.testing.assert_equal(np.nextafter(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)), np.nextafter(np.array(a, dtype=float_type), np.array(b, dtype=float_type)))

    @ignore_warning(category=RuntimeWarning, message='invalid value encountered')
    def testSpacing(self, float_type):
        with self.subTest(name='Subnormals'):
            for i in range(int(np.log2(float(ml_dtypes.finfo(float_type).smallest_subnormal))), int(np.log2(float(ml_dtypes.finfo(float_type).smallest_normal)))):
                power_of_two = float_type(2.0 ** i)
                distance = ml_dtypes.finfo(float_type).smallest_subnormal
                np.testing.assert_equal(np.spacing(power_of_two), distance)
                np.testing.assert_equal(np.spacing(-power_of_two), -distance)
        with self.subTest(name='Normals'):
            for i in range(int(np.log2(float(ml_dtypes.finfo(float_type).smallest_normal))), int(np.log2(float(ml_dtypes.finfo(float_type).max)))):
                power_of_two = float_type(2.0 ** i)
                distance = ml_dtypes.finfo(float_type).eps * power_of_two
                np.testing.assert_equal(np.spacing(power_of_two), distance)
                np.testing.assert_equal(np.spacing(-power_of_two), -distance)
        with self.subTest(name='NextAfter'):
            for x in FLOAT_VALUES[float_type]:
                x_float_type = float_type(x)
                spacing = np.spacing(x_float_type)
                toward = np.copysign(float_type(2.0 * np.abs(x) + 1), x_float_type)
                nextup = np.nextafter(x_float_type, toward)
                if np.isnan(spacing):
                    self.assertTrue(np.isnan(nextup - x_float_type))
                else:
                    np.testing.assert_equal(spacing, nextup - x_float_type)
        with self.subTest(name='NonFinite'):
            nan = float_type(float('nan'))
            np.testing.assert_equal(np.spacing(nan), np.spacing(np.float32(nan)))
            if dtype_has_inf(float_type):
                inf = float_type(float('inf'))
                np.testing.assert_equal(np.spacing(inf), np.spacing(np.float32(inf)))