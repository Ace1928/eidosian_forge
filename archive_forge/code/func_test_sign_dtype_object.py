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
def test_sign_dtype_object(self):
    foo = np.array([-0.1, 0, 0.1])
    a = np.sign(foo.astype(object))
    b = np.sign(foo)
    assert_array_equal(a, b)