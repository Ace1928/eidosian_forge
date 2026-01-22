from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class TestStringDiscovery:

    @pytest.mark.parametrize('obj', [object(), 1.2, 10 ** 43, None, 'string'], ids=['object', '1.2', '10**43', 'None', 'string'])
    def test_basic_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f'S{length}')
        assert np.array(obj, dtype='S').dtype == expected
        assert np.array([obj], dtype='S').dtype == expected
        arr = np.array(obj, dtype='O')
        assert np.array(arr, dtype='S').dtype == expected
        assert np.array(arr, dtype=type(expected)).dtype == expected
        assert arr.astype('S').dtype == expected
        assert arr.astype(type(np.dtype('S'))).dtype == expected

    @pytest.mark.parametrize('obj', [object(), 1.2, 10 ** 43, None, 'string'], ids=['object', '1.2', '10**43', 'None', 'string'])
    def test_nested_arrays_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f'S{length}')
        arr = np.array(obj, dtype='O')
        assert np.array([arr, arr], dtype='S').dtype == expected

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_unpack_first_level(self, arraylike):
        obj = np.array([None])
        obj[0] = np.array(1.2)
        length = len(str(obj[0]))
        expected = np.dtype(f'S{length}')
        obj = arraylike(obj)
        arr = np.array([obj], dtype='S')
        assert arr.shape == (1, 1)
        assert arr.dtype == expected