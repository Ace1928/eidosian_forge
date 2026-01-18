import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_warn_noclose():
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        it = np.nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
        del it
        assert len(sup.log) == 1