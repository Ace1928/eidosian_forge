from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testScalarArithmetic(self):
    xm = array(0, mask=1)
    with np.errstate(divide='ignore'):
        assert_((1 / array(0)).mask)
    assert_((1 + xm).mask)
    assert_((-xm).mask)
    assert_((-xm).mask)
    assert_(maximum(xm, xm).mask)
    assert_(minimum(xm, xm).mask)
    assert_(xm.filled().dtype is xm._data.dtype)
    x = array(0, mask=0)
    assert_(x.filled() == x._data)
    assert_equal(str(xm), str(masked_print_option))