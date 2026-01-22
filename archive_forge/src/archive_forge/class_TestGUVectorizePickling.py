import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
class TestGUVectorizePickling(TestCase):

    def test_pickle_gufunc_non_dyanmic(self):
        """Non-dynamic gufunc.
        """

        @guvectorize(['f8,f8[:]'], '()->()')
        def double(x, out):
            out[:] = x * 2
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs, double.gufunc_builder._sigs)
        self.assertTrue(cloned._frozen)
        cloned.disable_compile()
        self.assertTrue(cloned._frozen)
        self.assertPreciseEqual(double(0.5), cloned(0.5))
        arr = np.arange(10)
        self.assertPreciseEqual(double(arr), cloned(arr))

    def test_pickle_gufunc_dyanmic_null_init(self):
        """Dynamic gufunc w/o prepopulating before pickling.
        """

        @guvectorize('()->()', identity=1)
        def double(x, out):
            out[:] = x * 2
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs, double.gufunc_builder._sigs)
        self.assertFalse(cloned._frozen)
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        cloned(0.5, out=got)
        self.assertPreciseEqual(expect, got)
        arr = np.arange(10)
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)
        cloned(arr, out=got)
        self.assertPreciseEqual(expect, got)

    def test_pickle_gufunc_dynamic_initialized(self):
        """Dynamic gufunc prepopulated before pickling.

        Once unpickled, we disable compilation to verify that the gufunc
        compilation state is carried over.
        """

        @guvectorize('()->()', identity=1)
        def double(x, out):
            out[:] = x * 2
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        arr = np.arange(10)
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs, double.gufunc_builder._sigs)
        self.assertFalse(cloned._frozen)
        cloned.disable_compile()
        self.assertTrue(cloned._frozen)
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        cloned(0.5, out=got)
        self.assertPreciseEqual(expect, got)
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)
        cloned(arr, out=got)
        self.assertPreciseEqual(expect, got)