import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
@pytest.mark.parametrize('offset', [0, np.BUFSIZE // 2, int(1.5 * np.BUFSIZE)])
def test_reduce_casterrors(offset):
    value = 123
    arr = np.array([value] * offset + ['string'] + [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.array(-1, dtype=np.intp)
    count = sys.getrefcount(value)
    with pytest.raises(ValueError, match='invalid literal'):
        np.add.reduce(arr, dtype=np.intp, out=out, initial=None)
    assert count == sys.getrefcount(value)
    assert out[()] < value * offset