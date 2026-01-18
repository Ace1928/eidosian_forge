import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('casting', ['no', 'unsafe'])
def test_void_and_structured_with_subarray(self, casting):
    dtype = np.dtype([('foo', '<f4', (3, 2))])
    expected = casting == 'unsafe'
    assert np.can_cast('V4', dtype, casting=casting) == expected
    assert np.can_cast(dtype, 'V4', casting=casting) == expected