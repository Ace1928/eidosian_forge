from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testArrayMethods(self):
    a = array([1, 3, 2])
    assert_(eq(a.any(), a._data.any()))
    assert_(eq(a.all(), a._data.all()))
    assert_(eq(a.argmax(), a._data.argmax()))
    assert_(eq(a.argmin(), a._data.argmin()))
    assert_(eq(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4)))
    assert_(eq(a.compress([1, 0, 1]), a._data.compress([1, 0, 1])))
    assert_(eq(a.conj(), a._data.conj()))
    assert_(eq(a.conjugate(), a._data.conjugate()))
    m = array([[1, 2], [3, 4]])
    assert_(eq(m.diagonal(), m._data.diagonal()))
    assert_(eq(a.sum(), a._data.sum()))
    assert_(eq(a.take([1, 2]), a._data.take([1, 2])))
    assert_(eq(m.transpose(), m._data.transpose()))