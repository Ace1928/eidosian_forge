import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def test_vectorize_multiple_signatures(self):
    with captured_stdout():
        from numba import vectorize, int32, int64, float32, float64
        import numpy as np

        @vectorize([int32(int32, int32), int64(int64, int64), float32(float32, float32), float64(float64, float64)])
        def f(x, y):
            return x + y
        a = np.arange(6)
        result = f(a, a)
        self.assertIsInstance(result, np.ndarray)
        correct = np.array([0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(result, correct)
        a = np.linspace(0, 1, 6)
        result = f(a, a)
        self.assertIsInstance(result, np.ndarray)
        correct = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
        np.testing.assert_allclose(result, correct)
        a = np.arange(12).reshape(3, 4)
        result1 = f.reduce(a, axis=0)
        result2 = f.reduce(a, axis=1)
        result3 = f.accumulate(a)
        result4 = f.accumulate(a, axis=1)
        self.assertIsInstance(result1, np.ndarray)
        correct = np.array([12, 15, 18, 21])
        np.testing.assert_array_equal(result1, correct)
        self.assertIsInstance(result2, np.ndarray)
        correct = np.array([6, 22, 38])
        np.testing.assert_array_equal(result2, correct)
        self.assertIsInstance(result3, np.ndarray)
        correct = np.array([[0, 1, 2, 3], [4, 6, 8, 10], [12, 15, 18, 21]])
        np.testing.assert_array_equal(result3, correct)
        self.assertIsInstance(result4, np.ndarray)
        correct = np.array([[0, 1, 3, 6], [4, 9, 15, 22], [8, 17, 27, 38]])
        np.testing.assert_array_equal(result4, correct)