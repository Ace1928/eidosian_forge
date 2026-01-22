import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
class TestGlobals(unittest.TestCase):

    def check_global_ndarray(self, **jitargs):
        ctestfunc = jit(**jitargs)(global_ndarray_func)
        self.assertEqual(ctestfunc(1), 11)

    def test_global_ndarray(self):
        self.check_global_ndarray(forceobj=True)

    def test_global_ndarray_npm(self):
        self.check_global_ndarray(nopython=True)

    def check_global_complex_arr(self, **jitargs):
        ctestfunc = jit(**jitargs)(global_cplx_arr_copy)
        arr = np.zeros(len(cplx_X), dtype=np.complex128)
        ctestfunc(arr)
        np.testing.assert_equal(arr, cplx_X)

    def test_global_complex_arr(self):
        self.check_global_complex_arr(forceobj=True)

    def test_global_complex_arr_npm(self):
        self.check_global_complex_arr(nopython=True)

    def check_global_rec_arr(self, **jitargs):
        ctestfunc = jit(**jitargs)(global_rec_arr_copy)
        arr = np.zeros(rec_X.shape, dtype=x_dt)
        ctestfunc(arr)
        np.testing.assert_equal(arr, rec_X)

    def test_global_rec_arr(self):
        self.check_global_rec_arr(forceobj=True)

    def test_global_rec_arr_npm(self):
        self.check_global_rec_arr(nopython=True)

    def check_global_rec_arr_extract(self, **jitargs):
        ctestfunc = jit(**jitargs)(global_rec_arr_extract_fields)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        ctestfunc(arr1, arr2)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)

    def test_global_rec_arr_extract(self):
        self.check_global_rec_arr_extract(forceobj=True)

    def test_global_rec_arr_extract_npm(self):
        self.check_global_rec_arr_extract(nopython=True)

    def check_two_global_rec_arrs(self, **jitargs):
        ctestfunc = jit(**jitargs)(global_two_rec_arrs)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        arr3 = np.zeros(rec_Y.shape, dtype=np.int16)
        arr4 = np.zeros(rec_Y.shape, dtype=np.float64)
        ctestfunc(arr1, arr2, arr3, arr4)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)
        np.testing.assert_equal(arr3, rec_Y.c)
        np.testing.assert_equal(arr4, rec_Y.d)

    def test_two_global_rec_arrs(self):
        self.check_two_global_rec_arrs(forceobj=True)

    def test_two_global_rec_arrs_npm(self):
        self.check_two_global_rec_arrs(nopython=True)

    def test_global_module(self):
        res = global_module_func(5, 6)
        self.assertEqual(True, res)

    def test_global_record(self):
        x = np.recarray(1, dtype=x_dt)[0]
        x.a = 1
        res = global_record_func(x)
        self.assertEqual(True, res)
        x.a = 2
        res = global_record_func(x)
        self.assertEqual(False, res)

    def test_global_int_tuple(self):
        pyfunc = global_int_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_str_tuple(self):
        pyfunc = global_str_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_mixed_tuple(self):
        pyfunc = global_mixed_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_float_tuple(self):
        pyfunc = global_float_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_npy_int_tuple(self):
        pyfunc = global_npy_int_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_write_to_arr_in_tuple(self):
        for func in (global_write_to_arr_in_tuple, global_write_to_arr_in_mixed_tuple):
            jitfunc = njit(func)
            with self.assertRaises(errors.TypingError) as e:
                jitfunc()
            msg = 'Cannot modify readonly array of type:'
            self.assertIn(msg, str(e.exception))

    def test_global_npy_bool(self):
        pyfunc = global_npy_bool
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())