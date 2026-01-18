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
def test_object_nans(self):
    for i in range(1):
        x = np.array(float('nan'), object)
        y = 1.0
        z = np.array(float('nan'), object)
        assert_(np.minimum(x, y) == 1.0)
        assert_(np.minimum(z, y) == 1.0)