import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_scalar_wins2(self):
    with pytest.warns(DeprecationWarning, match='np.find_common_type'):
        res = np.find_common_type(['u4', 'i4', 'i4'], ['f4'])
    assert_(res == 'f8')