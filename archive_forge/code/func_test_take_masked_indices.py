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
def test_take_masked_indices(self):
    a = np.array((40, 18, 37, 9, 22))
    indices = np.arange(3)[None, :] + np.arange(5)[:, None]
    mindices = array(indices, mask=indices >= len(a))
    test = take(a, mindices, mode='clip')
    ctrl = array([[40, 18, 37], [18, 37, 9], [37, 9, 22], [9, 22, 22], [22, 22, 22]])
    assert_equal(test, ctrl)
    test = take(a, mindices)
    ctrl = array([[40, 18, 37], [18, 37, 9], [37, 9, 22], [9, 22, 40], [22, 40, 40]])
    ctrl[3, 2] = ctrl[4, 1] = ctrl[4, 2] = masked
    assert_equal(test, ctrl)
    assert_equal(test.mask, ctrl.mask)
    a = array((40, 18, 37, 9, 22), mask=(0, 1, 0, 0, 0))
    test = take(a, mindices)
    ctrl[0, 1] = ctrl[1, 0] = masked
    assert_equal(test, ctrl)
    assert_equal(test.mask, ctrl.mask)