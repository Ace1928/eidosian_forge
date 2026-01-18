import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize(['dtype', 'out_dtype'], [(np.bytes_, np.bool_), (np.str_, np.bool_), (np.dtype('S10,S9'), np.dtype('?,?'))])
def test_string_to_boolean_cast(dtype, out_dtype):
    """
    Currently, for `astype` strings are cast to booleans effectively by
    calling `bool(int(string)`. This is not consistent (see gh-9875) and
    will eventually be deprecated.
    """
    arr = np.array(['10', '10\x00\x00\x00', '0\x00\x00', '0'], dtype=dtype)
    expected = np.array([True, True, False, False], dtype=out_dtype)
    assert_array_equal(arr.astype(out_dtype), expected)