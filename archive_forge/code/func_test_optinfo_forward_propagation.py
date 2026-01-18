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
def test_optinfo_forward_propagation(self):
    a = array([1, 2, 2, 4])
    a._optinfo['key'] = 'value'
    assert_equal(a._optinfo['key'], (a == 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a != 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a > 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a >= 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a <= 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a + 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a - 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a * 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], (a / 2)._optinfo['key'])
    assert_equal(a._optinfo['key'], a[:2]._optinfo['key'])
    assert_equal(a._optinfo['key'], a[[0, 0, 2]]._optinfo['key'])
    assert_equal(a._optinfo['key'], np.exp(a)._optinfo['key'])
    assert_equal(a._optinfo['key'], np.abs(a)._optinfo['key'])
    assert_equal(a._optinfo['key'], array(a, copy=True)._optinfo['key'])
    assert_equal(a._optinfo['key'], np.zeros_like(a)._optinfo['key'])