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
def test_noshrinking(self):
    a = masked_array([1.0, 2.0, 3.0], mask=[False, False, False], shrink=False)
    b = a + 1
    assert_equal(b.mask, [0, 0, 0])
    a += 1
    assert_equal(a.mask, [0, 0, 0])
    b = a / 1.0
    assert_equal(b.mask, [0, 0, 0])
    a /= 1.0
    assert_equal(a.mask, [0, 0, 0])