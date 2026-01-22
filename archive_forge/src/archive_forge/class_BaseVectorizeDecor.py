import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
class BaseVectorizeDecor(object):
    target = None
    wrapper = None
    funcs = {'func1': sinc, 'func2': scaled_sinc, 'func3': vector_add}

    @classmethod
    def _run_and_compare(cls, func, sig, A, *args, **kwargs):
        if cls.wrapper is not None:
            func = cls.wrapper(func)
        numba_func = vectorize(sig, target=cls.target)(func)
        numpy_func = np.vectorize(func)
        result = numba_func(A, *args)
        gold = numpy_func(A, *args)
        np.testing.assert_allclose(result, gold, **kwargs)

    def test_1(self):
        sig = ['float64(float64)', 'float32(float32)']
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_2(self):
        sig = [float64(float64), float32(float32)]
        func = self.funcs['func1']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_3(self):
        sig = ['float64(float64, uint32)']
        func = self.funcs['func2']
        A = np.arange(100, dtype=np.float64)
        scale = np.uint32(3)
        self._run_and_compare(func, sig, A, scale, atol=1e-08)

    def test_4(self):
        sig = [int32(int32, int32), uint32(uint32, uint32), float32(float32, float32), float64(float64, float64)]
        func = self.funcs['func3']
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.float32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.int32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.uint32)
        self._run_and_compare(func, sig, A, A)