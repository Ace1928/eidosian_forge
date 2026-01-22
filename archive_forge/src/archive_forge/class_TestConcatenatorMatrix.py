import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
class TestConcatenatorMatrix:

    def test_matrix(self):
        a = [1, 2]
        b = [3, 4]
        ab_r = np.r_['r', a, b]
        ab_c = np.r_['c', a, b]
        assert_equal(type(ab_r), np.matrix)
        assert_equal(type(ab_c), np.matrix)
        assert_equal(np.array(ab_r), [[1, 2, 3, 4]])
        assert_equal(np.array(ab_c), [[1], [2], [3], [4]])
        assert_raises(ValueError, lambda: np.r_['rc', a, b])

    def test_matrix_scalar(self):
        r = np.r_['r', [1, 2], 3]
        assert_equal(type(r), np.matrix)
        assert_equal(np.array(r), [[1, 2, 3]])

    def test_matrix_builder(self):
        a = np.array([1])
        b = np.array([2])
        c = np.array([3])
        d = np.array([4])
        actual = np.r_['a, b; c, d']
        expected = np.bmat([[a, b], [c, d]])
        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))