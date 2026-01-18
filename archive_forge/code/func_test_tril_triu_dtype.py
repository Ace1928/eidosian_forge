from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_tril_triu_dtype():
    for c in np.typecodes['All']:
        if c == 'V':
            continue
        arr = np.zeros((3, 3), dtype=c)
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)
    arr = np.array([['2001-01-01T12:00', '2002-02-03T13:56'], ['2004-01-01T12:00', '2003-01-03T13:45']], dtype='datetime64')
    assert_equal(np.triu(arr).dtype, arr.dtype)
    assert_equal(np.tril(arr).dtype, arr.dtype)
    arr = np.zeros((3, 3), dtype='f4,f4')
    assert_equal(np.triu(arr).dtype, arr.dtype)
    assert_equal(np.tril(arr).dtype, arr.dtype)