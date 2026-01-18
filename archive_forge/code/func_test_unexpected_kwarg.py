import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_unexpected_kwarg(self):
    with assert_raises_regex(TypeError, 'nonsense'):
        np.array2string(np.array([1, 2, 3]), nonsense=None)