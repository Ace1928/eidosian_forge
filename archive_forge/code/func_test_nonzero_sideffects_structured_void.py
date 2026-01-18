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
def test_nonzero_sideffects_structured_void(self):
    arr = np.zeros(5, dtype='i1,i8,i8')
    assert arr.flags.aligned
    assert not arr['f2'].flags.aligned
    np.nonzero(arr)
    assert arr.flags.aligned
    np.count_nonzero(arr)
    assert arr.flags.aligned