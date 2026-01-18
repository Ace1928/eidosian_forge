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
def test_putmask(self):
    x = arange(6) + 1
    mx = array(x, mask=[0, 0, 0, 1, 1, 1])
    mask = [0, 0, 1, 0, 0, 1]
    xx = x.copy()
    putmask(xx, mask, 99)
    assert_equal(xx, [1, 2, 99, 4, 5, 99])
    mxx = mx.copy()
    putmask(mxx, mask, 99)
    assert_equal(mxx._data, [1, 2, 99, 4, 5, 99])
    assert_equal(mxx._mask, [0, 0, 0, 1, 1, 0])
    values = array([10, 20, 30, 40, 50, 60], mask=[1, 1, 1, 0, 0, 0])
    xx = x.copy()
    putmask(xx, mask, values)
    assert_equal(xx._data, [1, 2, 30, 4, 5, 60])
    assert_equal(xx._mask, [0, 0, 1, 0, 0, 0])
    mxx = mx.copy()
    putmask(mxx, mask, values)
    assert_equal(mxx._data, [1, 2, 30, 4, 5, 60])
    assert_equal(mxx._mask, [0, 0, 1, 1, 1, 0])
    mxx = mx.copy()
    mxx.harden_mask()
    putmask(mxx, mask, values)
    assert_equal(mxx, [1, 2, 30, 4, 5, 60])