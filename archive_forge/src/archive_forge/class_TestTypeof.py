import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
class TestTypeof(ValueTypingTestBase, TestCase):
    """
    Test typeof() and, implicitly, typing.Context.get_argument_type().
    """

    def test_number_values(self):
        """
        Test special.typeof() with scalar number values.
        """
        self.check_number_values(typeof)
        self.assertEqual(typeof(1), types.intp)
        self.assertEqual(typeof(-1), types.intp)
        self.assertEqual(typeof(2 ** 40), types.int64)
        self.assertEqual(typeof(2 ** 63), types.uint64)
        self.assertEqual(typeof(2 ** 63 - 1), types.int64)
        self.assertEqual(typeof(-2 ** 63), types.int64)

    def test_datetime_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_datetime_values(typeof)

    def test_timedelta_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_timedelta_values(typeof)

    def test_array_values(self):
        """
        Test special.typeof() with ndarray values.
        """

        def check(arr, ndim, layout, mutable, aligned):
            ty = typeof(arr)
            self.assertIsInstance(ty, types.Array)
            self.assertEqual(ty.ndim, ndim)
            self.assertEqual(ty.layout, layout)
            self.assertEqual(ty.mutable, mutable)
            self.assertEqual(ty.aligned, aligned)
        a1 = np.arange(10)
        check(a1, 1, 'C', True, True)
        a2 = np.arange(10).reshape(2, 5)
        check(a2, 2, 'C', True, True)
        check(a2.T, 2, 'F', True, True)
        a3 = np.arange(60)[::2].reshape((2, 5, 3))
        check(a3, 3, 'A', True, True)
        a4 = np.arange(1).reshape(())
        check(a4, 0, 'C', True, True)
        a4.flags.writeable = False
        check(a4, 0, 'C', False, True)
        a5 = a1.astype(a1.dtype.newbyteorder())
        with self.assertRaises(NumbaValueError) as raises:
            typeof(a5)
        self.assertIn('Unsupported array dtype: %s' % (a5.dtype,), str(raises.exception))
        with self.assertRaises(NumbaTypeError) as raises:
            masked_arr = np.ma.MaskedArray([1])
            typeof(masked_arr)
        self.assertIn(f'Unsupported array type: numpy.ma.MaskedArray', str(raises.exception))

    def test_structured_arrays(self):

        def check(arr, dtype, ndim, layout, aligned):
            ty = typeof(arr)
            self.assertIsInstance(ty, types.Array)
            self.assertEqual(ty.dtype, dtype)
            self.assertEqual(ty.ndim, ndim)
            self.assertEqual(ty.layout, layout)
            self.assertEqual(ty.aligned, aligned)
        dtype = np.dtype([('m', np.int32), ('n', 'S5')])
        rec_ty = numpy_support.from_struct_dtype(dtype)
        arr = np.empty(4, dtype=dtype)
        check(arr, rec_ty, 1, 'C', False)
        arr = np.recarray(4, dtype=dtype)
        check(arr, rec_ty, 1, 'C', False)
        dtype = np.dtype([('m', np.int32), ('n', 'S5')], align=True)
        rec_ty = numpy_support.from_struct_dtype(dtype)
        arr = np.empty(4, dtype=dtype)
        check(arr, rec_ty, 1, 'C', True)
        arr = np.recarray(4, dtype=dtype)
        check(arr, rec_ty, 1, 'C', True)

    def test_buffers(self):
        b = b'xx'
        ty = typeof(b)
        self.assertEqual(ty, types.Bytes(types.uint8, 1, 'C'))
        self.assertFalse(ty.mutable)
        ty = typeof(memoryview(b))
        self.assertEqual(ty, types.MemoryView(types.uint8, 1, 'C', readonly=True))
        self.assertFalse(ty.mutable)
        ty = typeof(array.array('i', [0, 1, 2]))
        self.assertEqual(ty, types.PyArray(types.int32, 1, 'C'))
        self.assertTrue(ty.mutable)
        b = bytearray(10)
        ty = typeof(b)
        self.assertEqual(ty, types.ByteArray(types.uint8, 1, 'C'))
        self.assertTrue(ty.mutable)

    def test_none(self):
        ty = typeof(None)
        self.assertEqual(ty, types.none)

    def test_ellipsis(self):
        ty = typeof(Ellipsis)
        self.assertEqual(ty, types.ellipsis)

    def test_str(self):
        ty = typeof('abc')
        self.assertEqual(ty, types.string)

    def test_slices(self):
        for args in [(1,), (1, 2), (1, 2, 1), (1, 2, None)]:
            v = slice(*args)
            self.assertIs(typeof(v), types.slice2_type)
        for args in [(1, 2, 2), (1, 2, -1), (None, None, -2)]:
            v = slice(*args)
            self.assertIs(typeof(v), types.slice3_type)

    def test_tuples(self):
        v = (1, 2)
        self.assertEqual(typeof(v), types.UniTuple(types.intp, 2))
        v = (1, (2.0, 3))
        self.assertEqual(typeof(v), types.Tuple((types.intp, types.Tuple((types.float64, types.intp)))))

    def test_lists(self):
        v = [1.0] * 100
        self.assertEqual(typeof(v), types.List(types.float64, reflected=True))
        bad_v = [{1: 3}]
        with self.assertRaises(ValueError) as raises:
            typeof(bad_v)
        self.assertIn('Cannot type list element type', str(raises.exception))

    def test_sets(self):
        v = set([1.0, 2.0, 3.0])
        self.assertEqual(typeof(v), types.Set(types.float64, reflected=True))
        v = frozenset(v)
        with self.assertRaises(ValueError) as raises:
            typeof(v)
        self.assertIn('Cannot determine Numba type of', str(raises.exception))

    def test_namedtuple(self):
        v = Point(1, 2)
        tp_point = typeof(v)
        self.assertEqual(tp_point, types.NamedUniTuple(types.intp, 2, Point))
        v = Point(1, 2.0)
        self.assertEqual(typeof(v), types.NamedTuple([types.intp, types.float64], Point))
        w = Rect(3, 4)
        tp_rect = typeof(w)
        self.assertEqual(tp_rect, types.NamedUniTuple(types.intp, 2, Rect))
        self.assertNotEqual(tp_rect, tp_point)
        self.assertNotEqual(tp_rect, types.UniTuple(tp_rect.dtype, tp_rect.count))

    def test_enum(self):
        tp_red = typeof(Color.red)
        self.assertEqual(tp_red, types.EnumMember(Color, types.intp))
        self.assertEqual(tp_red, typeof(Color.blue))
        tp_choc = typeof(Shake.chocolate)
        self.assertEqual(tp_choc, types.EnumMember(Shake, types.intp))
        self.assertEqual(tp_choc, typeof(Shake.mint))
        self.assertNotEqual(tp_choc, tp_red)
        tp_404 = typeof(RequestError.not_found)
        self.assertEqual(tp_404, types.IntEnumMember(RequestError, types.intp))
        self.assertEqual(tp_404, typeof(RequestError.internal_error))
        with self.assertRaises(ValueError) as raises:
            typeof(HeterogeneousEnum.red)
        self.assertEqual(str(raises.exception), 'Cannot type heterogeneous enum: got value types complex128, float64')

    def test_enum_class(self):
        tp_color = typeof(Color)
        self.assertEqual(tp_color, types.EnumClass(Color, types.intp))
        tp_shake = typeof(Shake)
        self.assertEqual(tp_shake, types.EnumClass(Shake, types.intp))
        self.assertNotEqual(tp_shake, tp_color)
        tp_shape = typeof(Shape)
        self.assertEqual(tp_shape, types.IntEnumClass(Shape, types.intp))
        tp_error = typeof(RequestError)
        self.assertEqual(tp_error, types.IntEnumClass(RequestError, types.intp))
        self.assertNotEqual(tp_error, tp_shape)
        with self.assertRaises(ValueError) as raises:
            typeof(HeterogeneousEnum)
        self.assertEqual(str(raises.exception), 'Cannot type heterogeneous enum: got value types complex128, float64')

    def test_dtype(self):
        dtype = np.dtype('int64')
        self.assertEqual(typeof(dtype), types.DType(types.int64))
        dtype = np.dtype([('m', np.int32), ('n', 'S5')])
        rec_ty = numpy_support.from_struct_dtype(dtype)
        self.assertEqual(typeof(dtype), types.DType(rec_ty))

    def test_dtype_values(self):
        self.assertEqual(typeof(np.int64), types.NumberClass(types.int64))
        self.assertEqual(typeof(np.float64), types.NumberClass(types.float64))
        self.assertEqual(typeof(np.int32), types.NumberClass(types.int32))
        self.assertEqual(typeof(np.int8), types.NumberClass(types.int8))

    def test_ctypes(self):
        ty_cos = typeof(c_cos)
        ty_sin = typeof(c_sin)
        self.assertIsInstance(ty_cos, types.ExternalFunctionPointer)
        self.assertEqual(ty_cos.sig.args, (types.float64,))
        self.assertEqual(ty_cos.sig.return_type, types.float64)
        self.assertEqual(ty_cos, ty_sin)
        self.assertNotEqual(ty_cos.get_pointer(c_cos), ty_sin.get_pointer(c_sin))

    @skip_unless_cffi
    def test_cffi(self):
        from numba.tests import cffi_usecases as mod
        mod.init()
        ty_cffi_cos = typeof(mod.cffi_cos)
        ty_cffi_sin = typeof(mod.cffi_sin)
        ty_cffi_boolean = typeof(mod.cffi_bool)
        self.assertIsInstance(ty_cffi_cos, types.ExternalFunctionPointer)
        self.assertEqual(ty_cffi_boolean.sig.return_type, types.boolean)
        self.assertEqual(ty_cffi_cos.sig.args, (types.float64,))
        self.assertEqual(ty_cffi_cos.sig.return_type, types.float64)
        self.assertEqual(ty_cffi_cos, ty_cffi_sin)
        ty_ctypes_cos = typeof(c_cos)
        self.assertNotEqual(ty_cffi_cos, ty_ctypes_cos)
        self.assertNotEqual(ty_cffi_cos.get_pointer(mod.cffi_cos), ty_cffi_sin.get_pointer(mod.cffi_sin))
        self.assertEqual(ty_cffi_cos.get_pointer(mod.cffi_cos), ty_ctypes_cos.get_pointer(c_cos))

    def test_custom(self):
        ty = typeof(Custom())
        self.assertEqual(ty, types.UniTuple(types.boolean, 42))

    def test_omitted_args(self):
        ty0 = typeof(OmittedArg(0.0))
        ty1 = typeof(OmittedArg(1))
        ty2 = typeof(OmittedArg(1.0))
        ty3 = typeof(OmittedArg(1.0))
        self.assertEqual(ty0, types.Omitted(0.0))
        self.assertEqual(ty1, types.Omitted(1))
        self.assertEqual(ty2, types.Omitted(1.0))
        self.assertEqual(len({ty0, ty1, ty2}), 3)
        self.assertEqual(ty3, ty2)

    def test_np_random(self):
        rng = np.random.default_rng()
        ty_rng = typeof(rng)
        ty_bitgen = typeof(rng.bit_generator)
        self.assertEqual(ty_rng, types.npy_rng)
        self.assertEqual(ty_bitgen, types.npy_bitgen)