import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_number_of_arguments(self):
    arr = np.empty((5,))
    for j in range(35):
        arrs = [arr] * j
        if j > 32:
            assert_raises(ValueError, np.broadcast, *arrs)
        else:
            mit = np.broadcast(*arrs)
            assert_equal(mit.numiter, j)