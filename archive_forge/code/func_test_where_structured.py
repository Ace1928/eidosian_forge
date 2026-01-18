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
def test_where_structured(self):
    dt = np.dtype([('a', int), ('b', int)])
    x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
    y = np.array((10, 20), dtype=dt)
    core = np.where([0, 1, 1], x, y)
    ma = np.where([0, 1, 1], x, y)
    assert_equal(core, ma)
    assert_equal(core.dtype, ma.dtype)