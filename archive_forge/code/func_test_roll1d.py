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
def test_roll1d(self):
    x = np.arange(10)
    xr = np.roll(x, 2)
    assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))