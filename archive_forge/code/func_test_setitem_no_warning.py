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
def test_setitem_no_warning(self):
    x = np.ma.arange(60).reshape((6, 10))
    index = (slice(1, 5, 2), [7, 5])
    value = np.ma.masked_all((2, 2))
    value._data[...] = np.inf
    x[index] = value
    x[...] = np.ma.masked
    x = np.ma.arange(3.0, dtype=np.float32)
    value = np.ma.array([2e+234, 1, 1], mask=[True, False, False])
    x[...] = value
    x[[0, 1, 2]] = value