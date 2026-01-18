import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_nonint_offsets(self):

    def make_dtype(off):
        return np.dtype({'names': ['A'], 'formats': ['i4'], 'offsets': [off]})
    assert_raises(TypeError, make_dtype, 'ASD')
    assert_raises(OverflowError, make_dtype, 2 ** 70)
    assert_raises(TypeError, make_dtype, 2.3)
    assert_raises(ValueError, make_dtype, -10)
    dt = make_dtype(np.uint32(0))
    np.zeros(1, dtype=dt)[0].item()