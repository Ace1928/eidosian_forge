from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class TestNested:

    def test_nested_simple(self):
        initial = [1.2]
        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]
        arr = np.array(nested, dtype='float64')
        assert arr.shape == (1,) * np.MAXDIMS
        with pytest.raises(ValueError):
            np.array([nested], dtype='float64')
        with pytest.raises(ValueError, match='.*would exceed the maximum'):
            np.array([nested])
        arr = np.array([nested], dtype=object)
        assert arr.dtype == np.dtype('O')
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() is initial

    def test_pathological_self_containing(self):
        l = []
        l.append(l)
        arr = np.array([l, l, l], dtype=object)
        assert arr.shape == (3,) + (1,) * (np.MAXDIMS - 1)
        arr = np.array([l, [None], l], dtype=object)
        assert arr.shape == (3, 1)

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_nested_arraylikes(self, arraylike):
        initial = arraylike(np.ones((1, 1)))
        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]
        with pytest.raises(ValueError, match='.*would exceed the maximum'):
            np.array(nested, dtype='float64')
        arr = np.array(nested, dtype=object)
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() == np.array(initial).item()

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_uneven_depth_ragged(self, arraylike):
        arr = np.arange(4).reshape((2, 2))
        arr = arraylike(arr)
        out = np.array([arr, [arr]], dtype=object)
        assert out.shape == (2,)
        assert out[0] is arr
        assert type(out[1]) is list
        with pytest.raises(ValueError):
            np.array([arr, [arr, arr]], dtype=object)

    def test_empty_sequence(self):
        arr = np.array([[], [1], [[1]]], dtype=object)
        assert arr.shape == (3,)
        with pytest.raises(ValueError):
            np.array([[], np.empty((0, 1))], dtype=object)

    def test_array_of_different_depths(self):
        arr = np.zeros((3, 2))
        mismatch_first_dim = np.zeros((1, 2))
        mismatch_second_dim = np.zeros((3, 3))
        dtype, shape = _discover_array_parameters([arr, mismatch_second_dim], dtype=np.dtype('O'))
        assert shape == (2, 3)
        dtype, shape = _discover_array_parameters([arr, mismatch_first_dim], dtype=np.dtype('O'))
        assert shape == (2,)
        res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype('O'))
        assert res[0] is arr
        assert res[1] is mismatch_first_dim