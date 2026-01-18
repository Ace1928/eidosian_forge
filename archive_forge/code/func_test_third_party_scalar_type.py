import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_third_party_scalar_type(self):
    from numpy.core._rational_tests import rational
    assert_raises(KeyError, np.sctype2char, rational)
    assert_raises(KeyError, np.sctype2char, rational(1))