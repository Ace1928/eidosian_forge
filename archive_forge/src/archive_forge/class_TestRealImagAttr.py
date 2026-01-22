import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
class TestRealImagAttr(MemoryLeakMixin, TestCase):

    def check_complex(self, pyfunc):
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size) + np.arange(size) * 10j
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))

    def test_complex_real(self):
        self.check_complex(array_real)

    def test_complex_imag(self):
        self.check_complex(array_imag)

    def check_number_real(self, dtype):
        pyfunc = array_real
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size, dtype=dtype)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        self.assertEqual(arr.data, pyfunc(arr).data)
        self.assertEqual(arr.data, cfunc(arr).data)
        real = cfunc(arr)
        self.assertNotEqual(arr[0, 0], 5)
        real[0, 0] = 5
        self.assertEqual(arr[0, 0], 5)

    def test_number_real(self):
        """
        Testing .real of non-complex dtypes
        """
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            self.check_number_real(dtype)

    def check_number_imag(self, dtype):
        pyfunc = array_imag
        cfunc = njit(pyfunc)
        size = 10
        arr = np.arange(size, dtype=dtype)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        arr = arr.reshape(2, 5)
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
        self.assertEqual(cfunc(arr).tolist(), np.zeros_like(arr).tolist())
        imag = cfunc(arr)
        with self.assertRaises(ValueError) as raises:
            imag[0] = 1
        self.assertEqual('assignment destination is read-only', str(raises.exception))

    def test_number_imag(self):
        """
        Testing .imag of non-complex dtypes
        """
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            self.check_number_imag(dtype)

    def test_record_real(self):
        rectyp = np.dtype([('real', np.float32), ('imag', np.complex64)])
        arr = np.zeros(3, dtype=rectyp)
        arr['real'] = np.random.random(arr.size)
        arr['imag'] = np.random.random(arr.size) * 1.3j
        self.assertIs(array_real(arr), arr)
        self.assertEqual(array_imag(arr).tolist(), np.zeros_like(arr).tolist())
        jit_array_real = njit(array_real)
        jit_array_imag = njit(array_imag)
        with self.assertRaises(TypingError) as raises:
            jit_array_real(arr)
        self.assertIn('cannot access .real of array of Record', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            jit_array_imag(arr)
        self.assertIn('cannot access .imag of array of Record', str(raises.exception))