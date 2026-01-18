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
def test_pickling_wstructured(self):
    a = array([(1, 1.0), (2, 2.0)], mask=[(0, 0), (0, 1)], dtype=[('a', int), ('b', float)])
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled, a)