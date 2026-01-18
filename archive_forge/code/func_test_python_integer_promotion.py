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
@pytest.mark.parametrize('val', [2, 2 ** 32, 2 ** 63, 2 ** 64, 2 * 100])
def test_python_integer_promotion(self, val):
    expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
    assert np.result_type(val, 0) == expected_dtype
    assert np.result_type(val, np.int8(0)) == expected_dtype