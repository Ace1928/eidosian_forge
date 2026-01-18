import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_shares_memory_api():
    x = np.zeros([4, 5, 6], dtype=np.int8)
    assert_equal(np.shares_memory(x, x), True)
    assert_equal(np.shares_memory(x, x.copy()), False)
    a = x[:, ::2, ::3]
    b = x[:, ::3, ::2]
    assert_equal(np.shares_memory(a, b), True)
    assert_equal(np.shares_memory(a, b, max_work=None), True)
    assert_raises(np.TooHardError, np.shares_memory, a, b, max_work=1)