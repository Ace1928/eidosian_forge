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
def test_sort_2d(self):
    a = masked_array([[8, 4, 1], [2, 0, 9]])
    a.sort(0)
    assert_equal(a, [[2, 0, 1], [8, 4, 9]])
    a = masked_array([[8, 4, 1], [2, 0, 9]])
    a.sort(1)
    assert_equal(a, [[1, 4, 8], [0, 2, 9]])
    a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
    a.sort(0)
    assert_equal(a, [[2, 0, 1], [8, 4, 9]])
    assert_equal(a._mask, [[0, 0, 0], [1, 0, 1]])
    a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
    a.sort(1)
    assert_equal(a, [[1, 4, 8], [0, 2, 9]])
    assert_equal(a._mask, [[0, 0, 1], [0, 0, 1]])
    a = masked_array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]], [[1, 2, 3], [7, 8, 9], [4, 5, 6]], [[7, 8, 9], [1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
    a[a % 4 == 0] = masked
    am = a.copy()
    an = a.filled(99)
    am.sort(0)
    an.sort(0)
    assert_equal(am, an)
    am = a.copy()
    an = a.filled(99)
    am.sort(1)
    an.sort(1)
    assert_equal(am, an)
    am = a.copy()
    an = a.filled(99)
    am.sort(2)
    an.sort(2)
    assert_equal(am, an)