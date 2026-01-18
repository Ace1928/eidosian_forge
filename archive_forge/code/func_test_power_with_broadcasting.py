import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_power_with_broadcasting(self):
    a2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2m = array(a2, mask=[[1, 0, 0], [0, 0, 1]])
    b1 = np.array([2, 4, 3])
    b2 = np.array([b1, b1])
    b2m = array(b2, mask=[[0, 1, 0], [0, 1, 0]])
    ctrl = array([[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]], mask=[[1, 1, 0], [0, 1, 1]])
    test = a2m ** b2m
    assert_equal(test, ctrl)
    assert_equal(test.mask, ctrl.mask)
    test = a2m ** b2
    assert_equal(test, ctrl)
    assert_equal(test.mask, a2m.mask)
    test = a2 ** b2m
    assert_equal(test, ctrl)
    assert_equal(test.mask, b2m.mask)
    ctrl = array([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]], mask=[[0, 1, 0], [0, 1, 0]])
    test = b1 ** b2m
    assert_equal(test, ctrl)
    assert_equal(test.mask, ctrl.mask)
    test = b2m ** b1
    assert_equal(test, ctrl)
    assert_equal(test.mask, ctrl.mask)