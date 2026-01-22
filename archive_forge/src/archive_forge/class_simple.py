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
class simple(np.ndarray):

    def __new__(subtype, shape):
        self = np.ndarray.__new__(subtype, shape, dtype=object)
        self.fill(0)
        return self