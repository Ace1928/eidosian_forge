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
def test_where_structured_masked(self):
    dt = np.dtype([('a', int), ('b', int)])
    x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
    ma = where([0, 1, 1], x, masked)
    expected = masked_where([1, 0, 0], x)
    assert_equal(ma.dtype, expected.dtype)
    assert_equal(ma, expected)
    assert_equal(ma.mask, expected.mask)