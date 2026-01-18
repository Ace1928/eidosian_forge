import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_signed_zeros(self):
    with np.errstate(all='ignore'):
        for t in [np.complex64, np.complex128]:
            data = (((0.0, -1.0), (0.0, 1.0), (-1.0, -0.0)), ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)), ((0.0, -1.0), (-0.0, -1.0), (1.0, 0.0)), ((0.0, -1.0), (-0.0, 1.0), (-1.0, 0.0)), ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)), ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)), ((-0.0, -1.0), (0.0, -1.0), (1.0, -0.0)), ((-0.0, 1.0), (0.0, -1.0), (-1.0, -0.0)))
            for cases in data:
                n = cases[0]
                d = cases[1]
                ex = cases[2]
                result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                assert_equal(result.real, ex[0])
                assert_equal(result.imag, ex[1])