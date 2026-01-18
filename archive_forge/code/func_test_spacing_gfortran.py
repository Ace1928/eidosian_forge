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
def test_spacing_gfortran():
    ref = {np.float64: [1.6940658945086007e-21, 2.220446049250313e-16, 1.1368683772161603e-13, 1.8189894035458565e-12], np.float32: [9.09494702e-13, 1.1920929e-07, 6.10351563e-05, 0.0009765625]}
    for dt, dec_ in zip([np.float32, np.float64], (10, 20)):
        x = np.array([1e-05, 1, 1000, 10500], dtype=dt)
        assert_array_almost_equal(np.spacing(x), ref[dt], decimal=dec_)