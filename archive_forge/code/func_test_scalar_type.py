import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_scalar_type(self):
    assert_equal(np.sctype2char(np.double), 'd')
    assert_equal(np.sctype2char(np.int_), 'l')
    assert_equal(np.sctype2char(np.str_), 'U')
    assert_equal(np.sctype2char(np.bytes_), 'S')