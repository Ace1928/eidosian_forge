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
def test_outer_bad_subclass():

    class BadArr1(np.ndarray):

        def __array_finalize__(self, obj):
            if self.ndim == 3:
                self.shape = self.shape + (1,)

        def __array_prepare__(self, obj, context=None):
            return obj

    class BadArr2(np.ndarray):

        def __array_finalize__(self, obj):
            if isinstance(obj, BadArr2):
                if self.shape[-1] == 1:
                    self.shape = self.shape[::-1]

        def __array_prepare__(self, obj, context=None):
            return obj
    for cls in [BadArr1, BadArr2]:
        arr = np.ones((2, 3)).view(cls)
        with assert_raises(TypeError) as a:
            np.add.outer(arr, [1, 2])
        arr = np.ones((2, 3)).view(cls)
        assert type(np.add.outer([1, 2], arr)) is cls