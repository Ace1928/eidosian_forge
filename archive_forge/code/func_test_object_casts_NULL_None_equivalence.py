import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('dtype', np.typecodes['All'])
def test_object_casts_NULL_None_equivalence(self, dtype):
    arr_normal = np.array([None] * 5)
    arr_NULLs = np.empty_like(arr_normal)
    ctypes.memset(arr_NULLs.ctypes.data, 0, arr_NULLs.nbytes)
    assert arr_NULLs.tobytes() == b'\x00' * arr_NULLs.nbytes
    try:
        expected = arr_normal.astype(dtype)
    except TypeError:
        with pytest.raises(TypeError):
            (arr_NULLs.astype(dtype),)
    else:
        assert_array_equal(expected, arr_NULLs.astype(dtype))