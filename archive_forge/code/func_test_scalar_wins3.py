import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_scalar_wins3(self):
    with pytest.warns(DeprecationWarning, match='np.find_common_type'):
        res = np.find_common_type(['u8', 'i8', 'i8'], ['f8'])
    assert_(res == 'f8')