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
@pytest.mark.parametrize('order', 'AKCF')
@pytest.mark.parametrize('data_order', 'CF')
def test_ravel_order(self, order, data_order):
    arr = np.ones((5, 10), order=data_order)
    arr[0, :] = 0
    mask = np.ones((10, 5), dtype=bool, order=data_order).T
    mask[0, :] = False
    x = array(arr, mask=mask)
    assert x._data.flags.fnc != x._mask.flags.fnc
    assert (x.filled(0) == 0).all()
    raveled = x.ravel(order)
    assert (raveled.filled(0) == 0).all()
    assert_array_equal(arr.ravel(order), x.ravel(order)._data)