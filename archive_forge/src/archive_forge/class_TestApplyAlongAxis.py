import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestApplyAlongAxis:

    def test_simple(self):
        a = np.ones((20, 10), 'd')
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_simple101(self):
        a = np.ones((10, 101), 'd')
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_3d(self):
        a = np.arange(27).reshape((3, 3, 3))
        assert_array_equal(apply_along_axis(np.sum, 0, a), [[27, 30, 33], [36, 39, 42], [45, 48, 51]])

    def test_preserve_subclass(self):

        def double(row):
            return row * 2

        class MyNDArray(np.ndarray):
            pass
        m = np.array([[0, 1], [2, 3]]).view(MyNDArray)
        expected = np.array([[0, 2], [4, 6]]).view(MyNDArray)
        result = apply_along_axis(double, 0, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)
        result = apply_along_axis(double, 1, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

    def test_subclass(self):

        class MinimalSubclass(np.ndarray):
            data = 1

        def minimal_function(array):
            return array.data
        a = np.zeros((6, 3)).view(MinimalSubclass)
        assert_array_equal(apply_along_axis(minimal_function, 0, a), np.array([1, 1, 1]))

    def test_scalar_array(self, cls=np.ndarray):
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(np.sum, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):

        def sum_to_0d(x):
            """ Sum x, returning a 0d array of the same class """
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(sum_to_0d, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))
        res = apply_along_axis(sum_to_0d, 1, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))

    def test_axis_insertion(self, cls=np.ndarray):

        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            return (x[::-1] * x[1:, None]).view(cls)
        a2d = np.arange(6 * 3).reshape((6, 3))
        actual = apply_along_axis(f1to2, 0, a2d)
        expected = np.stack([f1to2(a2d[:, i]) for i in range(a2d.shape[1])], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)
        actual = apply_along_axis(f1to2, 1, a2d)
        expected = np.stack([f1to2(a2d[i, :]) for i in range(a2d.shape[0])], axis=0).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)
        a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))
        actual = apply_along_axis(f1to2, 1, a3d)
        expected = np.stack([np.stack([f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])], axis=0) for j in range(a3d.shape[2])], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

    def test_subclass_preservation(self):

        class MinimalSubclass(np.ndarray):
            pass
        self.test_scalar_array(MinimalSubclass)
        self.test_0d_array(MinimalSubclass)
        self.test_axis_insertion(MinimalSubclass)

    def test_axis_insertion_ma(self):

        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            res = x[::-1] * x[1:, None]
            return np.ma.masked_where(res % 5 == 0, res)
        a = np.arange(6 * 3).reshape((6, 3))
        res = apply_along_axis(f1to2, 0, a)
        assert_(isinstance(res, np.ma.masked_array))
        assert_equal(res.ndim, 3)
        assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
        assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
        assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)

    def test_tuple_func1d(self):

        def sample_1d(x):
            return (x[1], x[0])
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):

        def never_call(x):
            assert_(False)
        a = np.empty((0, 0))
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        def empty_to_1(x):
            assert_(len(x) == 0)
            return 1
        a = np.empty((10, 0))
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    def test_with_iterable_object(self):
        d = np.array([[{1, 11}, {2, 22}, {3, 33}], [{4, 44}, {5, 55}, {6, 66}]])
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])
        assert_equal(actual, expected)
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))