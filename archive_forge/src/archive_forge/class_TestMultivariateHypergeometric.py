import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
class TestMultivariateHypergeometric:

    def setup_method(self):
        self.seed = 8675309

    def test_argument_validation(self):
        assert_raises(ValueError, random.multivariate_hypergeometric, 10, 4)
        assert_raises(ValueError, random.multivariate_hypergeometric, [2, 3, 4], -1)
        assert_raises(ValueError, random.multivariate_hypergeometric, [-1, 2, 3], 2)
        assert_raises(ValueError, random.multivariate_hypergeometric, [2, 3, 4], 10)
        assert_raises(ValueError, random.multivariate_hypergeometric, [], 1)
        assert_raises(ValueError, random.multivariate_hypergeometric, [999999999, 101], 5, 1, 'marginals')
        int64_info = np.iinfo(np.int64)
        max_int64 = int64_info.max
        max_int64_index = max_int64 // int64_info.dtype.itemsize
        assert_raises(ValueError, random.multivariate_hypergeometric, [max_int64_index - 100, 101], 5, 1, 'count')

    @pytest.mark.parametrize('method', ['count', 'marginals'])
    def test_edge_cases(self, method):
        random = Generator(MT19937(self.seed))
        x = random.multivariate_hypergeometric([0, 0, 0], 0, method=method)
        assert_array_equal(x, [0, 0, 0])
        x = random.multivariate_hypergeometric([], 0, method=method)
        assert_array_equal(x, [])
        x = random.multivariate_hypergeometric([], 0, size=1, method=method)
        assert_array_equal(x, np.empty((1, 0), dtype=np.int64))
        x = random.multivariate_hypergeometric([1, 2, 3], 0, method=method)
        assert_array_equal(x, [0, 0, 0])
        x = random.multivariate_hypergeometric([9, 0, 0], 3, method=method)
        assert_array_equal(x, [3, 0, 0])
        colors = [1, 1, 0, 1, 1]
        x = random.multivariate_hypergeometric(colors, sum(colors), method=method)
        assert_array_equal(x, colors)
        x = random.multivariate_hypergeometric([3, 4, 5], 12, size=3, method=method)
        assert_array_equal(x, [[3, 4, 5]] * 3)

    @pytest.mark.parametrize('nsample', [8, 25, 45, 55])
    @pytest.mark.parametrize('method', ['count', 'marginals'])
    @pytest.mark.parametrize('size', [5, (2, 3), 150000])
    def test_typical_cases(self, nsample, method, size):
        random = Generator(MT19937(self.seed))
        colors = np.array([10, 5, 20, 25])
        sample = random.multivariate_hypergeometric(colors, nsample, size, method=method)
        if isinstance(size, int):
            expected_shape = (size,) + colors.shape
        else:
            expected_shape = size + colors.shape
        assert_equal(sample.shape, expected_shape)
        assert_((sample >= 0).all())
        assert_((sample <= colors).all())
        assert_array_equal(sample.sum(axis=-1), np.full(size, fill_value=nsample, dtype=int))
        if isinstance(size, int) and size >= 100000:
            assert_allclose(sample.mean(axis=0), nsample * colors / colors.sum(), rtol=0.001, atol=0.005)

    def test_repeatability1(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5, method='count')
        expected = np.array([[2, 1, 2], [2, 1, 2], [1, 1, 3], [2, 0, 3], [2, 1, 2]])
        assert_array_equal(sample, expected)

    def test_repeatability2(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 50, size=5, method='marginals')
        expected = np.array([[9, 17, 24], [7, 13, 30], [9, 15, 26], [9, 17, 24], [12, 14, 24]])
        assert_array_equal(sample, expected)

    def test_repeatability3(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 12, size=5, method='marginals')
        expected = np.array([[2, 3, 7], [5, 3, 4], [2, 5, 5], [5, 3, 4], [1, 5, 6]])
        assert_array_equal(sample, expected)