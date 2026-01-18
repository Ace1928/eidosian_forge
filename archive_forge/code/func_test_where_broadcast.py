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
def test_where_broadcast(self):
    x = np.arange(9).reshape(3, 3)
    y = np.zeros(3)
    core = np.where([1, 0, 1], x, y)
    ma = where([1, 0, 1], x, y)
    assert_equal(core, ma)
    assert_equal(core.dtype, ma.dtype)