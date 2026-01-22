import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
class TestHstack:

    def test_non_iterable(self):
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with pytest.raises(TypeError, match='arrays to stack must be'):
            hstack((np.arange(3) for _ in range(2)))
        with pytest.raises(TypeError, match='arrays to stack must be'):
            hstack(map(lambda x: x, np.ones((3, 2))))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting='unsafe', dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting='safe', dtype=np.int64)