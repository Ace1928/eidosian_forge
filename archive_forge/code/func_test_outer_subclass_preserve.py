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
@pytest.mark.parametrize('arr', [np.arange(2), np.matrix([0, 1]), np.matrix([[0, 1], [2, 5]])])
def test_outer_subclass_preserve(arr):

    class foo(np.ndarray):
        pass
    actual = np.multiply.outer(arr.view(foo), arr.view(foo))
    assert actual.__class__.__name__ == 'foo'