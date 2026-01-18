import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_reduce_alignment(self):
    a = np.zeros(2, dtype=[('a', np.int32), ('b', np.float64)])
    a['a'] = -1
    assert_equal(a['b'].sum(), 0)