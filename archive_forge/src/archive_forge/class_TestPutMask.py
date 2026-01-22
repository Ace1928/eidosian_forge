import sys
import pytest
import numpy as np
from numpy.testing import (
class TestPutMask:

    @pytest.mark.parametrize('dtype', list(np.typecodes['All']) + ['i,O'])
    def test_simple(self, dtype):
        if dtype.lower() == 'm':
            dtype += '8[ns]'
        vals = np.arange(1001).astype(dtype=dtype)
        mask = np.random.randint(2, size=1000).astype(bool)
        arr = np.zeros(1000, dtype=vals.dtype)
        zeros = arr.copy()
        np.putmask(arr, mask, vals)
        assert_array_equal(arr[mask], vals[:len(mask)][mask])
        assert_array_equal(arr[~mask], zeros[~mask])

    @pytest.mark.parametrize('dtype', list(np.typecodes['All'])[1:] + ['i,O'])
    @pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
    def test_empty(self, dtype, mode):
        arr = np.zeros(1000, dtype=dtype)
        arr_copy = arr.copy()
        mask = np.random.randint(2, size=1000).astype(bool)
        np.put(arr, mask, [])
        assert_array_equal(arr, arr_copy)