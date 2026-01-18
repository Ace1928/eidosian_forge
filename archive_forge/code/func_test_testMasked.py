from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testMasked(self):
    xx = arange(6)
    xx[1] = masked
    assert_(str(masked) == '--')
    assert_(xx[1] is masked)
    assert_equal(filled(xx[1], 0), 0)