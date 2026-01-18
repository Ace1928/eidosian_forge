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
def test_ufunc_override_not_implemented(self):

    class A:

        def __array_ufunc__(self, *args, **kwargs):
            return NotImplemented
    msg = "operand type(s) all returned NotImplemented from __array_ufunc__(<ufunc 'negative'>, '__call__', <*>): 'A'"
    with assert_raises_regex(TypeError, fnmatch.translate(msg)):
        np.negative(A())
    msg = "operand type(s) all returned NotImplemented from __array_ufunc__(<ufunc 'add'>, '__call__', <*>, <object *>, out=(1,)): 'A', 'object', 'int'"
    with assert_raises_regex(TypeError, fnmatch.translate(msg)):
        np.add(A(), object(), out=1)