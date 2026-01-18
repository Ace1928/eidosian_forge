import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_rjust(self):
    assert_(issubclass(self.A.rjust(10).dtype.type, np.bytes_))
    C = self.A.rjust([10, 20])
    assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])
    C = self.A.rjust(20, b'#')
    assert_(np.all(C.startswith(b'#')))
    assert_array_equal(C.endswith(b'#'), [[False, True], [False, False], [False, False]])
    C = np.char.rjust(b'FOO', [[10, 20], [15, 8]])
    tgt = [[b'       FOO', b'                 FOO'], [b'            FOO', b'     FOO']]
    assert_(issubclass(C.dtype.type, np.bytes_))
    assert_array_equal(C, tgt)