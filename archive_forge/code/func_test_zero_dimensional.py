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
def test_zero_dimensional(self):
    arr_0d = np.array(1)
    ret = np.tensordot(arr_0d, arr_0d, ([], []))
    assert_array_equal(ret, arr_0d)