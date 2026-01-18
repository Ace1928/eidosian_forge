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
def test_toflex(self):
    data = arange(10)
    record = data.toflex()
    assert_equal(record['_data'], data._data)
    assert_equal(record['_mask'], data._mask)
    data[[0, 1, 2, -1]] = masked
    record = data.toflex()
    assert_equal(record['_data'], data._data)
    assert_equal(record['_mask'], data._mask)
    ndtype = [('i', int), ('s', '|S3'), ('f', float)]
    data = array([(i, s, f) for i, s, f in zip(np.arange(10), 'ABCDEFGHIJKLM', np.random.rand(10))], dtype=ndtype)
    data[[0, 1, 2, -1]] = masked
    record = data.toflex()
    assert_equal(record['_data'], data._data)
    assert_equal(record['_mask'], data._mask)
    ndtype = np.dtype('int, (2,3)float, float')
    data = array([(i, f, ff) for i, f, ff in zip(np.arange(10), np.random.rand(10), np.random.rand(10))], dtype=ndtype)
    data[[0, 1, 2, -1]] = masked
    record = data.toflex()
    assert_equal_records(record['_data'], data._data)
    assert_equal_records(record['_mask'], data._mask)